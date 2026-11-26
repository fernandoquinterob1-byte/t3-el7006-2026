"""One table with every experiment of the project, for the summary notebook.

Each experiment is one entry of ``EXPERIMENTS``: a loader that returns its
per-fold test scores (F1, precision, recall, FP, FN) and the text the summary
notebook shows.  To add an experiment, write its loader and append an entry;
``notebooks/11_resumen_experimentos.ipynb`` picks it up automatically.

All loaders read caches or exported NPZ files: nothing is trained here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import ensemble_size, hybrid_threshold, hysteresis, probability_mix
from .metrics import EVENT_IOU_THRESHOLD, calculate_metrics_simple
from .report import EVALUATIONS, FOLDS, METHODS, method_path

SCORE_COLUMNS = ["F1 (%)", "precisión (%)", "recall (%)", "FP", "FN"]
REFERENCE = ("Métodos base", "Deep Ensemble")


@dataclass
class Experiment:
    key: str
    title: str
    notebook: str
    question: str
    what: str
    conclusion: str
    loader: Callable[[], pd.DataFrame] | None  # None = analysis without F1 variants (text only)


# ------------------------------------------------------------------ loaders

def _scores_from_ious(ious: np.ndarray) -> dict[str, float]:
    curve = calculate_metrics_simple(ious.tolist(), thresholds=np.array([EVENT_IOU_THRESHOLD]))
    return {"F1 (%)": 100 * float(curve["f1"][0]), "precisión (%)": 100 * float(curve["precision"][0]),
            "recall (%)": 100 * float(curve["recall"][0]), "FP": int(curve["false_positives"][0]),
            "FN": int(curve["false_negatives"][0])}


def load_methods() -> pd.DataFrame:
    """Every uncertainty method with its original threshold (exported test NPZ)."""
    rows = []
    for method in METHODS:
        for evaluation in EVALUATIONS:
            for fold in FOLDS:
                with np.load(method_path(method, evaluation, fold), allow_pickle=False) as data:
                    ious = data["event_iou_values"]
                rows.append({"evaluación": evaluation, "fold": fold, "variante": method, **_scores_from_ious(ious)})
    return pd.DataFrame(rows)


def load_tuned_threshold() -> pd.DataFrame:
    baselines = hybrid_threshold.run_all()["baselines"]
    rows = []
    for _, row in baselines.iterrows():
        rows.append({"evaluación": row["evaluación"], "fold": row["fold"],
                     "variante": f"{row['método']}: umbral ajustado",
                     **{c: row[f"{c} ajustado"] for c in SCORE_COLUMNS}})
    return pd.DataFrame(rows)


def _rule_scores(scores: pd.DataFrame, base: str = "original 0.425") -> pd.DataFrame:
    part = scores[(scores["base"] == base) & (scores["regla"] != "Deep Ensemble")]
    return part.rename(columns={"regla": "variante"})[["evaluación", "fold", "variante", *SCORE_COLUMNS]]


def load_joint_rules() -> pd.DataFrame:
    return _rule_scores(hybrid_threshold.run_all()["scores"])


def load_single_error() -> pd.DataFrame:
    return _rule_scores(hybrid_threshold.run_single())


def load_probability_mix() -> pd.DataFrame:
    scores = probability_mix.run_all()["scores"]
    part = scores[~scores["método"].isin(["Deep Ensemble", "Neural Process"])]
    return part.rename(columns={"método": "variante"})[["evaluación", "fold", "variante", *SCORE_COLUMNS]]


def load_hysteresis() -> pd.DataFrame:
    """Post-processing variants (all three methods; the original rows are already in "Métodos base")."""
    scores = hysteresis.run_all()["scores"]
    part = scores[scores["variante"] != "Original"].copy()
    part["variante"] = part["método"] + ": " + part["variante"].str.lower()
    return part[["evaluación", "fold", "variante", *SCORE_COLUMNS]]


def load_ensemble_size() -> pd.DataFrame:
    """Deep Ensemble with 1 to 4 members (5 members is the reference)."""
    sizes = ensemble_size.run_all()
    part = sizes[sizes["miembros"] < sizes["miembros"].max()].copy()
    part["variante"] = "Deep Ensemble con " + part["miembros"].astype(str) + " miembro" + \
        np.where(part["miembros"] > 1, "s", "")
    return part[["evaluación", "fold", "variante", *SCORE_COLUMNS]]


# --------------------------------------------------------------- registry

EXPERIMENTS: list[Experiment] = [
    Experiment(
        key="Métodos base",
        title="Métodos de incertidumbre (umbral original 0.425)",
        notebook="01–08",
        question="¿Qué método de incertidumbre detecta mejor los husos y señala mejor sus errores?",
        what="SEED y cinco métodos de incertidumbre sobre SEED, con el postprocesado y la F1 originales.",
        conclusion="El Deep Ensemble tiene el mejor F1 (83.6%) y la mejor calibración; Neural Process tiene la MI "
                   "más útil para señalar errores. El Deep Ensemble es la **referencia** de todos los experimentos "
                   "siguientes.",
        loader=load_methods,
    ),
    Experiment(
        key="Umbral ajustado",
        title="Ajustar el umbral de probabilidad sin incertidumbre",
        notebook="09 (sección 1)",
        question="¿Cuánto cambia el F1 si solo se mueve el umbral de 0.425, elegido en validación?",
        what="Grilla de umbrales 0.25–0.65 elegida con la F1 de validación, para SEED, Deep Ensemble y Neural Process.",
        conclusion="Casi nada (−0.2 a +0.2 puntos): el umbral original ya está cerca del óptimo. Es el punto de "
                   "partida justo para las reglas con incertidumbre.",
        loader=load_tuned_threshold,
    ),
    Experiment(
        key="Reglas FP + FN",
        title="Reglas con incertidumbre que corrigen FP y FN a la vez",
        notebook="09",
        question="¿Se puede usar la MI de Neural Process para borrar detecciones dudosas y recuperar husos perdidos?",
        what="Sobre las detecciones del Deep Ensemble: cortes fijos, umbral adaptativo con la MI de NP y "
             "regresión logística por evento, con y sin la condición de no bajar la precisión.",
        conclusion="Ninguna regla mejora el F1 (0 a −0.3 puntos). La MI de NP repite lo que ya dice la probabilidad "
                   "del Deep Ensemble en las detecciones, y no separa los casi detectados (AUROC 0.55).",
        loader=load_joint_rules,
    ),
    Experiment(
        key="Solo FP o solo FN",
        title="Corregir un solo tipo de error",
        notebook="09 (sección 6)",
        question="¿Funciona mejor corregir solo los FP o solo los FN, con menos parámetros?",
        what="Las mismas reglas, pero cada una solo borra detecciones (FP) o solo agrega casi detectados (FN).",
        conclusion="El F1 no sube, pero sirve como control: **solo FP** sube la precisión +1.5 puntos y "
                   "**solo FN** sube el recall hasta +1.65, con el F1 casi igual. Mueve errores de un lado al otro.",
        loader=load_single_error,
    ),
    Experiment(
        key="Mezcla de probabilidades",
        title="Mezclar las probabilidades de Deep Ensemble y Neural Process",
        notebook="10",
        question="¿Promediar las dos probabilidades da una mejor detección? ¿Cuánto confiar en cada una?",
        what="Grilla de pesos (con y sin umbral) y regresión logística sobre los logits (con y sin MI), "
             "todo elegido en validación.",
        conclusion="Empatan con el Deep Ensemble (−0.07 a +0.04 puntos). La logística confía ~8% en NP y da la "
                   "mejor calibración del proyecto (ECE 0.0026 vs 0.0037).",
        loader=load_probability_mix,
    ),
    Experiment(
        key="Histéresis",
        title="Postprocesado con histéresis y duración mínima",
        notebook="12",
        question="¿Se recuperan los husos que el modelo ve por menos de 0.3 s cambiando solo la regla final?",
        what="Dos umbrales (uno alto para empezar un evento, uno bajo para continuarlo), otra duración mínima, "
             "y las dos juntas, sin tocar las redes; para Deep Ensemble, SEED y Neural Process.",
        conclusion="No mejora el F1 (−0.27 a +0.01 puntos contra el postprocesado original de cada método). Recuperar "
                   "husos cortos trae casi la misma cantidad de picos cortos de ruido: con umbral bajo 0.35, +16 husos "
                   "y +23 FP por fold. La regla original ya está bien ajustada.",
        loader=load_hysteresis,
    ),
    Experiment(
        key="Tamaño del ensemble",
        title="¿Más semillas en el Deep Ensemble?",
        notebook="13",
        question="¿Mejoraría el Deep Ensemble con más miembros (semillas)?",
        what="F1 del promedio de cada subconjunto de 1 a 5 de los SEED ya entrenados, y una curva F1(M) = a − b/M "
             "para extrapolar a más semillas. Sin entrenar.",
        conclusion="Ganancias decrecientes: +0.46, +0.14, +0.06 y +0.02 al pasar de 1 a 5 miembros. La extrapolación "
                   "da ~83.68 con 10 semillas y un techo de ~83.77: entrenar más semillas iguales casi no ayuda.",
        loader=load_ensemble_size,
    ),
    Experiment(
        key="Análisis de FN",
        title="¿Cómo son los husos que el modelo no detecta?",
        notebook="14",
        question="¿Por qué ningún experimento baja los FN sin subir los FP? ¿Los husos perdidos son débiles, raros, "
                 "de ciertos sujetos o anotaciones dudosas?",
        what="Cada huso anotado del test se clasifica según qué tan cerca estuvo el Deep Ensemble de verlo, y se mide "
             "en la señal EEG su duración, amplitud y contraste en la banda sigma (11–16 Hz), frente a los TP, los FP y "
             "ventanas de fondo. Análisis sin variantes de F1.",
        conclusion="El 48% de los FN son **invisibles** (probabilidad < 0.25): husos reales pero **débiles y cortos** "
                   "(37% menos amplitud sigma que un TP, aunque 2.3 veces la del fondo), que el modelo descarta **con "
                   "seguridad** (incertidumbre tan baja como en los aciertos). Donde el modelo duda, ni la señal "
                   "distingue lo anotado de lo no anotado (AUROC 0.48–0.54), y un tercio de los FP se parece a un huso "
                   "típico. El límite del F1 es la **ambigüedad de los husos débiles en la anotación**.",
        loader=None,
    ),
    Experiment(
        key="Dinámica de incertidumbre",
        title="Curvas de incertidumbre sobre la probabilidad, por duración y frente al SEED base",
        notebook="15",
        question="¿Cómo se mueve u(t) junto con p(t)? ¿Cambia la incertidumbre con la duración del huso? ¿La "
                 "incertidumbre del DE y del NP es la misma que la del SEED base (suma cero) o rescata algo?",
        what="Con los eventos del SEED base (TP, FN, FP y ventanas TN), se promedian las curvas de p y u alineadas al "
             "inicio y fin de cada evento; se compara u por duración (corto < 0.6 s ≤ mediano < 0.9 s ≤ largo) y se "
             "hacen pair plots de u SEED vs u del DE y MI del DE y NP, con AUROC de errores vs aciertos. Sin variantes "
             "de F1.",
        conclusion="La incertidumbre total es casi una copia de p; solo la MI tiene información propia. La duda se "
                   "concentra en los **bordes** del huso (inicio y fin), y los eventos **cortos** son inciertos sean "
                   "aciertos o errores. Para los **FP es suma cero**: la duda del SEED ya los separa (AUROC 0.88) y la "
                   "MI no agrega. Para los **FN sí se rescata algo**: combinar u SEED con la MI del DE y del NP sube "
                   "la AUROC FN vs TP de 0.62 a 0.71, también dentro de cada duración. Pero eso se mide sobre husos "
                   "anotados; en el registro completo la señal se diluye en el fondo, por eso no subió el F1.",
        loader=None,
    ),
    Experiment(
        key="Fine-tuning con incertidumbre (piloto)",
        title="Fine-tuning de SEED con pérdida pesada por incertidumbre (eval 0, fold 0)",
        notebook="— (src/eeg_uq/uncertainty_finetune.py; resultados en outputs/uncertainty_finetune/)",
        question="¿Usar la incertidumbre del Deep Ensemble para guiar un fine-tuning corto mejora la predicción de SEED?",
        what="Desde el SEED original (semilla 42) se entrenan hasta 3 épocas con learning rate 1e-5. **F**: la pérdida de "
             "cada instante pesa más donde la MI del DE es alta (hasta ×3) y menos donde la incertidumbre aleatoria es "
             "alta (hasta ×0.5). **A (control)**: el mismo fine-tuning con pesos 1. La época se elige con el F1 de "
             "validación; test se mide una vez. A se repite con 4 semillas de entrenamiento (42, 1, 2, 3; cambia "
             "el orden de los batches, el punto de partida es el mismo). Solo un fold: no entra en la tabla comparativa.",
        conclusion="Test F1: SEED original 82.32 → **F 82.89** (+0.57). **A** con 4 semillas: 83.33 / 83.97 / 83.33 / "
                   "84.09, media **83.68** (+1.36), y las 4 superan al SEED original y al mejor de los 5 SEED "
                   "reentrenados (81.84–83.53). Unas pocas épocas más a learning rate bajo sí ayudan de forma "
                   "consistente en este fold (sobre todo bajan los FP: 279 → ~170–220). La variante con incertidumbre "
                   "queda **por debajo de las 4 semillas de A**: subir el peso de las zonas con MI alta empuja a "
                   "detectar más (recall 88%, FP 247) y pierde precisión. Con estos pesos, la incertidumbre no ayuda "
                   "al fine-tuning. **Extensión a la eval 0 completa (A, semilla 42, folds 0–4)**: test F1 "
                   "82.32→83.33, 81.76→81.15, 83.05→83.35, 83.08→83.64, 81.73→82.80; media SEED **82.39 → A 82.85** "
                   "(+0.46, mejora en 4 de 5 folds). Pero la media de los 5 SEED reentrenados es 82.91: el fine-tuning "
                   "deja al modelo en un nivel típico de reentrenar, no por encima. El fold 0 fue el caso más "
                   "favorable. El fine-tuning tiende a equilibrar precisión y recall (en folds con precisión alta, "
                   "la baja y sube el recall; en folds con recall alto, al revés).",
        loader=None,
    ),
]


# --------------------------------------------------------------- tables

def all_scores() -> pd.DataFrame:
    """Per-fold test scores of every variant of every experiment."""
    frames = []
    for experiment in EXPERIMENTS:
        if experiment.loader is None:
            continue
        frame = experiment.loader().assign(experimento=experiment.key)
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    scores["evaluación"] = scores["evaluación"].astype(int)
    scores["fold"] = scores["fold"].astype(int)
    return scores[["experimento", "variante", "evaluación", "fold", *SCORE_COLUMNS]]


def compare(scores: pd.DataFrame, reference: tuple[str, str] = REFERENCE) -> pd.DataFrame:
    """Mean over folds and paired change vs the reference (Deep Ensemble, original threshold)."""
    ref = scores[(scores["experimento"] == reference[0]) & (scores["variante"] == reference[1])]
    ref = ref.set_index(["evaluación", "fold"])
    rows = []
    for (experiment, variant), part in scores.groupby(["experimento", "variante"], sort=False):
        part = part.set_index(["evaluación", "fold"])
        delta = (part["F1 (%)"] - ref["F1 (%)"]).dropna()
        try:
            p_value = wilcoxon(delta).pvalue if (delta != 0).any() else np.nan
        except ValueError:
            p_value = np.nan
        rows.append({
            "experimento": experiment, "variante": variant,
            "F1 (%)": part["F1 (%)"].mean(), "± F1": part["F1 (%)"].std(),
            "precisión (%)": part["precisión (%)"].mean(), "recall (%)": part["recall (%)"].mean(),
            "FP": part["FP"].mean(), "FN": part["FN"].mean(),
            "Δ F1": delta.mean(),
            "IC95 Δ F1": 1.96 * delta.std() / np.sqrt(len(delta)) if len(delta) > 1 else np.nan,
            "Δ precisión": (part["precisión (%)"] - ref["precisión (%)"]).mean(),
            "Δ recall": (part["recall (%)"] - ref["recall (%)"]).mean(),
            "folds que mejoran F1": int((delta > 0).sum()), "folds": len(delta),
            "p Wilcoxon": p_value,
        })
    return pd.DataFrame(rows)
