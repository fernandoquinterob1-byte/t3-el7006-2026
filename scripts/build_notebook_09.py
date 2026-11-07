"""Build notebooks/09_umbral_mixto.ipynb (cells only; run_nb.py executes it)."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "09_umbral_mixto.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Umbral mixto: Deep Ensemble detecta, Neural Process corrige

**Idea.** Usar la probabilidad del Deep Ensemble para decidir dónde hay un huso y la duda de Neural Process (su MI) para corregir sus errores: borrar detecciones dudosas (falsos positivos) y recuperar husos que se le pasaron (falsos negativos).

**Datos.** Evaluaciones 0, 1 y 2, cinco folds cada una (15 folds). Se leen los `validation_predictions.npz` y `test_predictions.npz` ya guardados: no se entrena ni se vuelve a inferir. En cada fold, todo umbral o modelo se elige con **validación** y se prueba **una sola vez** en **test**. El código está en `src/eeg_uq/hybrid_threshold.py`.

**Pasos**

1. **Punto de partida justo**: ajustar el umbral de probabilidad sin incertidumbre.
2. **Lista de eventos**: las detecciones del Deep Ensemble y los tramos *casi detectados* (probabilidad ≥ 0.25 que no llegaron a detección), con sus medidas de los dos métodos.
3. **¿Vale la pena mezclar?** ¿La MI de Neural Process dice algo que la probabilidad del Deep Ensemble no dice ya?
4. **Tres reglas**: (1) cortes fijos, (2) umbral que baja donde Neural Process duda, (3) regresión logística por evento.
5. **Resultado en test** con la F1 oficial (IoU ≥ 0.2, el mismo emparejamiento del notebook original).

**Regla de F1.** Con F1 ≈ 83%, borrar una detección sube el F1 solo si más del ~58% de lo borrado es FP; agregar un tramo lo sube solo si más del ~42% de lo agregado es huso real (F1/2).
""")

code("""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

_candidates = [Path.cwd(), *Path.cwd().parents]
REPO_ROOT = next((p for p in _candidates if (p / 'configs' / 'default.yaml').exists()), Path.cwd())
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))
from eeg_uq import hybrid_threshold as ht
from eeg_uq import report

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 250)


def show(figure):
    display(figure)
    plt.close(figure)

REFRESH = False  # True recalcula desde los NPZ (~3 minutos)
results = ht.run_all(refresh=REFRESH)
scores, separation, counts, events, params = (results[k] for k in ('scores', 'separation', 'counts', 'events', 'params'))
""")

md("""
## 1. Punto de partida: ajustar el umbral sin incertidumbre

El postprocesado (`process_spindle_simple`) **solo usa el umbral de duración** (0.425): la máscara del umbral de detección (0.5) se recibe pero no se usa. Por eso se ajusta un solo número, en una grilla de 0.25 a 0.65 elegida con la F1 de validación.
""")

code("""
baselines = results['baselines']
for method in ['SEED', 'Deep Ensemble', 'Neural Process']:
    print(method)
    display(report.by_evaluation(baselines[baselines['método'] == method],
                                 ['umbral elegido', 'F1 (%) original', 'F1 (%) ajustado', 'precisión (%) original',
                                  'precisión (%) ajustado', 'recall (%) original', 'recall (%) ajustado']))
""")

md("""
## 2. Lista de eventos

Por fold de test: cuántas detecciones y cuántos *casi detectados* hay, qué parte de los casi detectados es un huso real, y qué parte de los husos perdidos (FN) está entre ellos (el resto tiene probabilidad < 0.25 y ninguna regla de este notebook puede recuperarlos).
""")

code("""
display(report.by_evaluation(counts[counts['base'] == 'original 0.425'],
                             ['detecciones (test)', 'casi detectados (test)', '% reales en casi detectados',
                              '% de los FN que están en casi detectados']))
""")

md("""
## 3. ¿Vale la pena mezclar?

AUROC en test para encontrar el error de cada grupo: en las **detecciones**, los FP; en los **casi detectados**, los husos reales. La logística se entrena en validación con las medidas del Deep Ensemble solo, o con las del Deep Ensemble más las de Neural Process. *ρ* es la correlación de Spearman entre la MI de los dos métodos en los mismos eventos.

Criterio fijado antes de mirar el test: si agregar Neural Process no sube el AUROC al menos ~0.02, la mezcla no vale la pena.
""")

code("""
SEPARATION = ['eventos', 'positivos', 'azar AUPRC', 'ρ Spearman MI DE vs MI NP', 'AUROC p media DE', 'AUROC MI DE', 'AUROC MI NP',
              'AUROC logística solo DE', 'AUROC logística DE + NP', 'AUPRC logística solo DE', 'AUPRC logística DE + NP']
base_sep = separation[separation['base'] == 'original 0.425']
for group in ['detecciones: TP vs FP', 'casi detectados: huso real vs no']:
    print(group)
    display(report.by_evaluation(base_sep[base_sep['grupo'] == group], SEPARATION))
""")

md("""
### Gráficos de cajón: MI de Neural Process por tipo de evento

Percentil de la MI dentro del fold (0 = la menor, 1 = la mayor) en los eventos de test. Izquierda: detecciones (TP vs FP). Derecha: casi detectados (huso real vs no). La línea punteada en 0.5 es el azar; la roja, la media.

Comparar con el gráfico de los notebooks 06 y 07: allí los FN se miden en el **tramo real** del huso, que el modelo no conoce al predecir. Aquí se mide en el tramo que el modelo sí ve.
""")

code("""
ev = events[events['base'] == 'original 0.425'].copy()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
fig.suptitle('MI de Neural Process (percentil dentro del fold), eventos de test')
panels = [
    (axes[0], ev[ev['candidato'] == 0], {1: 'TP', 0: 'FP'}, ['#2a78d6', '#eb6834'], 'Detecciones del Deep Ensemble'),
    (axes[1], ev[ev['candidato'] == 1], {1: 'huso real', 0: 'no es huso'}, ['#e34948', '#a0a0a0'], 'Casi detectados'),
]
for ax, part, names, colors, title in panels:
    part = part.assign(percentil=part.groupby(['evaluación', 'fold'])['MI NP'].rank(pct=True))
    values = [part.loc[part['real'] == k, 'percentil'] for k in names]
    box = ax.boxplot(values, tick_labels=list(names.values()), patch_artist=True, showfliers=False,
                     medianprops={'color': 'black'}, showmeans=True, meanline=True, meanprops={'color': 'red', 'linestyle': '-'})
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0.5, color=report.CHANCE, linestyle=':')
    ax.set_title(title)
    ax.set_ylabel('percentil de la MI de NP')
show(fig)
""")

md("""
## 4 y 5. Reglas y resultado en test

Cada regla se aplica sobre dos bases del Deep Ensemble: el umbral **original** (0.425) y el **ajustado en validación** (paso 1). La versión *(precisión ≥ base)* elige los cortes con la condición de que la precisión de validación no baje de la del Deep Ensemble.

- **Regla 1, cortes fijos**: borrar una detección si su probabilidad < t_p **y** la MI de NP > τ_FP; agregar un casi detectado si la MI de NP > τ_FN.
- **Regla 2, umbral adaptativo**: p ≥ umbral − β · percentil(MI de NP), por timestep, seguido del postprocesado normal.
- **Regla 3, logística**: un modelo para las detecciones y otro para los casi detectados (la MI significa cosas opuestas en cada grupo), cada uno con su corte de probabilidad elegido en validación.

Δ es la diferencia **pareada** por fold contra el Deep Ensemble de la misma base; *p Wilcoxon* compara esas 15 diferencias con cero.
""")

code("""
display(ht.summary(scores))
""")

code("""
rules = [r for r in scores['regla'].unique() if r != 'Deep Ensemble']
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True, constrained_layout=True)
fig.suptitle('Cambio de F1 por fold contra el Deep Ensemble (puntos; cada punto es un fold de test)')
for ax, base in zip(axes, scores['base'].unique()):
    part = scores[scores['base'] == base].set_index(['evaluación', 'fold'])
    ref = part[part['regla'] == 'Deep Ensemble']['F1 (%)']
    for i, rule in enumerate(rules):
        delta = part[part['regla'] == rule]['F1 (%)'] - ref
        ax.scatter(np.full(len(delta), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(delta)), delta, s=14, color='#4a3aa7', alpha=0.7)
        ax.hlines(delta.mean(), i - 0.3, i + 0.3, color='red')
    ax.axhline(0, color=report.CHANCE, linestyle=':')
    ax.set_xticks(range(len(rules)), [r.replace(' (precisión ≥ base)', '\\n(prec. ≥ base)').replace(': ', ':\\n') for r in rules],
                  rotation=90, fontsize=8)
    ax.set_title(f'Base: {base}')
    ax.set_ylabel('Δ F1 (puntos)')
show(fig)
""")

md("""
### Validación vs test

Para ver si las reglas no funcionan o si lo que ganan en validación no se mantiene en test: F1 de la regla logística DE + NP en validación (los mismos datos con que se eligió) frente a test.
""")

code("""
rows = []
for evaluation in report.EVALUATIONS:
    for fold in report.FOLDS:
        val = ht.load_pair(evaluation, fold, 'validation')
        val_events = ht.event_table(val, ht.DURATION_THRESHOLD)
        fitted = ht.rule_logistic(val_events, ht.DE_FEATURES + ht.NP_FEATURES, None)
        base = ht.event_scores(val['y_true'], ht.process_masks(val['de']['probability'], ht.DURATION_THRESHOLD))['F1 (%)']
        rule = ht.event_scores(val['y_true'], ht.edited_masks(val['y_true'].shape, val_events, ht.apply_logistic(val_events, fitted)))['F1 (%)']
        test = scores[(scores['base'] == 'original 0.425') & (scores['evaluación'] == evaluation) & (scores['fold'] == fold)].set_index('regla')['F1 (%)']
        rows.append({'evaluación': evaluation, 'fold': fold, 'Δ F1 validación': rule - base,
                     'Δ F1 test': test['Regla 3: logística DE + NP'] - test['Deep Ensemble']})
in_sample = pd.DataFrame(rows)
display(report.by_evaluation(in_sample, ['Δ F1 validación', 'Δ F1 test']))
""")

md("""
### Parámetros elegidos

Mediana entre folds de los cortes elegidos en validación (base original). `inf` en τ significa que la regla prefirió no borrar o no agregar nada.
""")

code("""
chosen = params[params['base'] == 'original 0.425']
keys = [c for c in chosen.columns if c.startswith(('regla', 'umbral'))]
display(chosen[keys].replace(np.inf, np.nan).median().to_frame('mediana').round(3))
print('Folds donde la regla 1 no borra nada:', int(np.isinf(chosen['regla 1: tau_fp']).sum()), '/ 15;',
      'no agrega nada:', int(np.isinf(chosen['regla 1: tau_fn']).sum()), '/ 15')
""")

md("""
## 6. Corregir un solo tipo de error

Las mismas reglas, pero cada una hace **una sola cosa**: o solo borra detecciones (FP), o solo agrega casi detectados (FN). Tienen menos parámetros que las reglas combinadas, así que se sobreajustan menos a validación. Base: umbral original 0.425.
""")

code("""
single = ht.run_single(refresh=REFRESH)
display(ht.summary(single).droplevel(0))
""")

md("""
**Resultado.** Ninguna sube el F1, pero funcionan como un control: **solo FP** sube la precisión hasta +1.5 puntos (logística del Deep Ensemble, 85.9%) con el F1 igual, y **solo FN** sube el recall hasta +1.65 (84.6%) con el F1 algo menor. Por cada huso recuperado entran casi dos FP nuevos. La MI de Neural Process no mejora la versión con solo el Deep Ensemble.
""")

md("""
## Hallazgos

Promedio de los 15 folds de test.

1. **Ajustar el umbral de probabilidad casi no cambia nada.** Deep Ensemble 83.57 → 83.59, SEED 82.72 → 82.88, Neural Process 83.34 → 83.14. El umbral original ya está cerca del óptimo, porque la probabilidad está bien calibrada.
2. **En las detecciones, la MI de Neural Process repite lo que ya dice la probabilidad del Deep Ensemble.** Separa los FP con AUROC 0.83, pero la probabilidad del Deep Ensemble sola llega a 0.86, y juntarlas en la logística da 0.861 contra 0.859 (+0.002, lejos del +0.02 exigido). Las MI de los dos métodos están muy correlacionadas en esos eventos (ρ ≈ 0.73).
3. **En los casi detectados, ninguna medida sirve.** La MI de Neural Process baja de AUROC 0.71 (medida sobre el tramo real del huso, notebooks 06 y 07) a ~0.55 sobre los tramos que el modelo realmente ve. Solo ~35% de esos tramos son husos reales, menos que el 42% necesario para que agregar suba el F1, y nada permite elegir un subconjunto mejor.
4. **Solo un tercio de los husos perdidos está entre los casi detectados.** El resto tiene probabilidad < 0.25: ninguna regla basada en estos tramos puede recuperarlos.
5. **Ninguna regla mejora el F1 en test.** Los cambios van de 0 a −0.4 puntos. La logística DE + NP sube el F1 +0.77 puntos en validación, pero baja −0.29 en test: lo que gana es sobreajuste a los datos con que se eligió. Los cortes elegidos para agregar casi detectados quedan en ~0.42, justo el F1/2 que predice la teoría, pero casi ningún tramo lo supera en test.
6. **Lo que sí se puede hacer es cambiar F1 por precisión.** Las versiones *(precisión ≥ base)* suben la precisión ~0.5–0.7 puntos con una pérdida de F1 de ~0.1. Es útil si un FP cuesta más que un FN.

**Conclusión.** Mezclar los dos métodos con un umbral no mejora la detección de SEED. La incertidumbre de los dos métodos sigue sirviendo para **señalar** detecciones dudosas y mandarlas a revisión (notebook 07, sección 8: el 20% más incierto contiene ~60% de los FP), pero no para **decidir** automáticamente mejor que la probabilidad del Deep Ensemble. Para recuperar husos perdidos haría falta una señal que aparezca donde la probabilidad es baja; en estos datos, la MI no la da.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": ".venv (3.13.12)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.12", "file_extension": ".py", "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3}, "nbconvert_exporter": "python",
                          "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"cell-{index:02d}"
NB.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("written", NB)
