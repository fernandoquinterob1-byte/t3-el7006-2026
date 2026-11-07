"""Mix the Deep Ensemble and Neural Process probabilities, timestep by timestep.

Two ways to decide how much to trust each method, both fitted on the
validation split of each fold and scored once on its test split:

* **Grid**: ``p = (1 - w) · p_DE + w · p_NP`` with the weight ``w`` (and,
  optionally, the detection threshold) chosen by the event F1 on validation.
* **Logistic regression (stacking)**: ``p = σ(a · logit p_DE + b · logit p_NP + c)``
  fitted by log-loss on validation timesteps; optionally with the MI of both
  methods and their interactions, so the weight can change with the doubt.
  Its detection threshold is then chosen by the event F1 on validation.

The mixed probability goes through the legacy post-processing and the legacy
event F1 (IoU >= 0.2), exactly like every other method of the project.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .hybrid_threshold import event_scores, load_pair, process_masks
from .method_summary import positive_class_ece
from .metrics import DURATION_THRESHOLD
from .report import CACHE, EVALUATIONS, FOLDS

CACHE_FILE = CACHE / "probability_mix.pkl"
CACHE_VERSION = 1
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
THRESHOLDS = np.round(np.arange(0.25, 0.651, 0.025), 3)
FIT_SAMPLES = 400_000  # validation timesteps used to fit the logistic models
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def _features(pair: dict, with_mi: bool) -> np.ndarray:
    """Per-timestep features, shape (records, time, k)."""
    l_de, l_np = _logit(pair["de"]["probability"]), _logit(pair["np"]["probability"])
    columns = [l_de, l_np]
    if with_mi:
        m_de = np.log10(pair["de"]["mi"] + EPS)
        m_np = np.log10(pair["np"]["mi"] + EPS)
        # Interactions let the weight of each probability change with each MI.
        columns += [m_de, m_np, l_de * m_de, l_de * m_np, l_np * m_de, l_np * m_np]
    return np.stack(columns, axis=-1)


FEATURE_NAMES = {
    False: ["logit p_DE", "logit p_NP"],
    True: ["logit p_DE", "logit p_NP", "log MI_DE", "log MI_NP",
           "logit p_DE × log MI_DE", "logit p_DE × log MI_NP", "logit p_NP × log MI_DE", "logit p_NP × log MI_NP"],
}


def fit_stacking(pair: dict, with_mi: bool, seed: int = 0):
    """Logistic regression on a random subset of valid validation timesteps."""
    valid = pair["de"]["valid_mask"].astype(bool)
    x = _features(pair, with_mi)[valid]
    y = pair["y_true"][valid]
    rng = np.random.default_rng(seed)
    if len(y) > FIT_SAMPLES:
        index = rng.choice(len(y), FIT_SAMPLES, replace=False)
        x, y = x[index], y[index]
    if with_mi:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    else:
        # No scaling: the coefficients stay readable as "trust in each logit".
        model = LogisticRegression(C=1.0, max_iter=3000)
    model.fit(x, y)
    return model


def stacking_probability(model, pair: dict, with_mi: bool) -> np.ndarray:
    x = _features(pair, with_mi)
    shape = x.shape[:2]
    return model.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(shape)


def grid_probability(pair: dict, weight: float) -> np.ndarray:
    return (1 - weight) * pair["de"]["probability"] + weight * pair["np"]["probability"]


def _best_threshold(pair: dict, probability: np.ndarray) -> tuple[float, float]:
    f1 = {t: event_scores(pair["y_true"], process_masks(probability, t))["F1 (%)"] for t in THRESHOLDS}
    best = max(f1, key=lambda t: (f1[t], -abs(t - DURATION_THRESHOLD)))
    return float(best), f1[best]


def _calibration(pair: dict, probability: np.ndarray) -> dict[str, float]:
    valid = pair["de"]["valid_mask"].astype(bool)
    p, y = np.clip(probability[valid], EPS, 1 - EPS), pair["y_true"][valid].astype(float)
    return {"NLL": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "ECE": positive_class_ece(p, y)}


def fold_analysis(evaluation: int, fold: int) -> dict[str, pd.DataFrame]:
    tag = {"evaluación": evaluation, "fold": fold}
    val, test = load_pair(evaluation, fold, "validation"), load_pair(evaluation, fold, "test")
    rows, params, curve = [], {**tag}, []

    def score(name: str, test_probability: np.ndarray, threshold: float, val_f1: float):
        rows.append({**tag, "método": name, "umbral": threshold, "F1 val (%)": val_f1,
                     **event_scores(test["y_true"], process_masks(test_probability, threshold)),
                     **_calibration(test, test_probability)})

    # References: each method alone, original threshold.
    for name, key in (("Deep Ensemble", "de"), ("Neural Process", "np")):
        val_f1 = event_scores(val["y_true"], process_masks(val[key]["probability"], DURATION_THRESHOLD))["F1 (%)"]
        score(name, test[key]["probability"], DURATION_THRESHOLD, val_f1)

    # Grid, threshold fixed at 0.425: choose only the weight.
    val_f1_by_w = {}
    for w in WEIGHTS:
        val_f1_by_w[w] = event_scores(val["y_true"], process_masks(grid_probability(val, w), DURATION_THRESHOLD))["F1 (%)"]
        test_f1 = event_scores(test["y_true"], process_masks(grid_probability(test, w), DURATION_THRESHOLD))["F1 (%)"]
        curve.append({**tag, "peso NP": float(w), "F1 val (%)": val_f1_by_w[w], "F1 test (%)": test_f1})
    w_fixed = max(val_f1_by_w, key=lambda w: (val_f1_by_w[w], -abs(w - 0.5)))
    score("Grilla: peso", grid_probability(test, w_fixed), DURATION_THRESHOLD, val_f1_by_w[w_fixed])
    params["grilla: peso NP"] = float(w_fixed)

    # Grid, weight and threshold together.
    best = (-1.0, None, None)
    for w in WEIGHTS:
        probability = grid_probability(val, w)
        for t in THRESHOLDS:
            f1 = event_scores(val["y_true"], process_masks(probability, t))["F1 (%)"]
            if f1 > best[0]:
                best = (f1, float(w), float(t))
    score("Grilla: peso + umbral", grid_probability(test, best[1]), best[2], best[0])
    params["grilla + umbral: peso NP"], params["grilla + umbral: umbral"] = best[1], best[2]

    # Logistic stacking, without and with MI.
    for with_mi, name in ((False, "Logística: p_DE + p_NP"), (True, "Logística: p + MI")):
        model = fit_stacking(val, with_mi)
        threshold, val_f1 = _best_threshold(val, stacking_probability(model, val, with_mi))
        score(name, stacking_probability(model, test, with_mi), threshold, val_f1)
        params[f"{name}: umbral"] = threshold
        estimator = model[-1] if hasattr(model, "steps") else model
        for feature, coef in zip(FEATURE_NAMES[with_mi], estimator.coef_[0]):
            params[f"{name}: coef {feature}"] = float(coef)
        params[f"{name}: intercepto"] = float(estimator.intercept_[0])

    return {"scores": pd.DataFrame(rows), "params": pd.DataFrame([params]), "curve": pd.DataFrame(curve)}


def run_all(refresh: bool = False, progress=print) -> dict[str, pd.DataFrame]:
    if CACHE_FILE.is_file() and not refresh:
        with CACHE_FILE.open("rb") as handle:
            stored = pickle.load(handle)
        if stored.get("version") == CACHE_VERSION:
            return stored["results"]
    parts: dict[str, list] = {}
    for evaluation in EVALUATIONS:
        for fold in FOLDS:
            for name, value in fold_analysis(evaluation, fold).items():
                parts.setdefault(name, []).append(value)
            progress(f"evaluación {evaluation}, fold {fold} listo")
    results = {name: pd.concat(values, ignore_index=True) for name, values in parts.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("wb") as handle:
        pickle.dump({"version": CACHE_VERSION, "results": results}, handle)
    return results


def summary(scores: pd.DataFrame, reference: str = "Deep Ensemble") -> pd.DataFrame:
    """Mean ± std over the folds and the paired change vs ``reference``."""
    from scipy.stats import wilcoxon

    base = scores[scores["método"] == reference].set_index(["evaluación", "fold"])
    rows = []
    for name, part in scores.groupby("método", sort=False):
        part = part.set_index(["evaluación", "fold"])
        delta = part["F1 (%)"] - base["F1 (%)"]
        try:
            p_value = wilcoxon(delta).pvalue if (delta != 0).any() else np.nan
        except ValueError:
            p_value = np.nan
        rows.append({
            "método": name,
            "F1 (%)": f"{part['F1 (%)'].mean():.2f} ± {part['F1 (%)'].std():.2f}",
            "precisión (%)": f"{part['precisión (%)'].mean():.2f} ± {part['precisión (%)'].std():.2f}",
            "recall (%)": f"{part['recall (%)'].mean():.2f} ± {part['recall (%)'].std():.2f}",
            "FP": round(part["FP"].mean(), 1), "FN": round(part["FN"].mean(), 1),
            "NLL": round(part["NLL"].mean(), 4), "ECE": round(part["ECE"].mean(), 4),
            "Δ F1 (puntos)": round(delta.mean(), 2),
            "Δ precisión (puntos)": round((part["precisión (%)"] - base["precisión (%)"]).mean(), 2),
            "folds que mejoran F1": f"{int((delta > 0).sum())}/{len(delta)}",
            "p Wilcoxon": round(p_value, 4) if not np.isnan(p_value) else np.nan,
        })
    return pd.DataFrame(rows).set_index("método")
