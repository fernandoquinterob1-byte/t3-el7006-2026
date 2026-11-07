"""Build notebooks/10_mezcla_probabilidades.ipynb (run_nb.py executes it)."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "10_mezcla_probabilidades.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Mezcla de probabilidades: Deep Ensemble + Neural Process

**Idea.** En cada instante, cada método da una probabilidad de huso. En vez de usar solo una, se **mezclan las dos** en una sola probabilidad, que después pasa por el postprocesado de siempre (umbral, duración mínima) y la F1 oficial (IoU ≥ 0.2).

**Dos formas de decidir cuánto confiar en cada método**

| | Grilla | Regresión logística (*stacking*) |
|---|---|---|
| Fórmula | p = (1 − w) · p_DE + w · p_NP | p = σ(a · logit p_DE + b · logit p_NP + c) |
| Cómo se elige | Se prueban pesos w de 0 a 1 (de 0.05 en 0.05) y gana el mejor F1 de validación | Un modelo aprende a, b y c con los timesteps de validación (minimiza log-loss) |
| Qué optimiza | El F1 por eventos | Que la probabilidad quede bien calibrada; después se elige el umbral por F1 |
| Variante | Elegir también el umbral | Agregar la MI de cada método, para que el peso cambie según cuánto duda cada uno |

**Protocolo.** Evaluaciones 0, 1 y 2, cinco folds cada una (15 folds). En cada fold, pesos, coeficientes y umbrales se eligen con **validación** y se prueban **una sola vez** en **test**. No se entrena ninguna red: se usan los `.npz` guardados. Código: `src/eeg_uq/probability_mix.py`.
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
from eeg_uq import probability_mix as pm
from eeg_uq import report

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 250)


def show(figure):
    display(figure)
    plt.close(figure)

REFRESH = False  # True recalcula desde los NPZ (~6 minutos)
results = pm.run_all(refresh=REFRESH)
scores, params, curve = results['scores'], results['params'], results['curve']
""")

md("""
## 1. Comparación en test

Δ es la diferencia **pareada** por fold contra el Deep Ensemble solo; *p Wilcoxon* compara esas 15 diferencias con cero. NLL y ECE miden la calibración de la probabilidad mezclada (menor es mejor).
""")

code("""
display(pm.summary(scores))
""")

code("""
print('F1 por evaluación')
f1 = scores.pivot_table(index=['evaluación', 'fold'], columns='método', values='F1 (%)', sort=False).reset_index()
display(report.by_evaluation(f1, [c for c in f1.columns if c not in ('evaluación', 'fold')]))
""")

md("""
## 2. Grilla: ¿qué peso conviene?

F1 promedio de los 15 folds para cada peso de Neural Process (0 = solo Deep Ensemble, 1 = solo Neural Process), con el umbral fijo en 0.425. A la izquierda, en validación (donde se elige); a la derecha, en test. Debajo, cuántos folds de test mejoran contra el Deep Ensemble solo y qué peso eligió la validación en cada fold.
""")

code("""
mean_curve = curve.groupby('peso NP')[['F1 val (%)', 'F1 test (%)']].mean()
test_by_fold = curve.pivot_table(index=['evaluación', 'fold'], columns='peso NP', values='F1 test (%)')
improves = (test_by_fold.sub(test_by_fold[0.0], axis=0) > 0).sum()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
for ax, column, title in ((axes[0], 'F1 val (%)', 'Validación'), (axes[1], 'F1 test (%)', 'Test')):
    ax.plot(mean_curve.index, mean_curve[column], marker='o', color='#4a3aa7')
    ax.axhline(mean_curve[column].iloc[0], color=report.CHANCE, linestyle=':', label='solo Deep Ensemble')
    ax.set_xlabel('peso de Neural Process (w)')
    ax.set_ylabel('F1 por evento (%)')
    ax.set_title(title)
    ax.legend()
axes[2].bar(params['grilla: peso NP'].value_counts().sort_index().index,
            params['grilla: peso NP'].value_counts().sort_index().values, width=0.04, color='#e87ba4')
axes[2].set_xlabel('peso elegido en validación')
axes[2].set_ylabel('folds')
axes[2].set_title('Peso elegido por fold (grilla, umbral 0.425)')
axes[2].set_xlim(-0.05, 1.05)
show(fig)
display(pd.DataFrame({'F1 val (%)': mean_curve['F1 val (%)'], 'F1 test (%)': mean_curve['F1 test (%)'],
                      'folds de test que mejoran': improves}).T.round(2))
""")

md("""
## 3. Regresión logística: ¿cuánto confía en cada método?

Coeficientes por fold (media, mediana, mínimo y máximo). En la versión **sin MI**, los coeficientes se leen directo: es el peso que recibe el logit de cada método. En la versión **con MI**, las entradas están estandarizadas: un término *logit × MI* negativo significa que el modelo le baja el peso a esa probabilidad cuando esa MI es alta.
""")

code("""
coefs = params[[c for c in params.columns if 'coef' in c or 'intercepto' in c]]
display(coefs.describe().T[['mean', '50%', 'min', 'max']].round(3))
ratio = params['Logística: p_DE + p_NP: coef logit p_NP'] / params['Logística: p_DE + p_NP: coef logit p_DE']
print(f'Peso relativo de Neural Process frente al Deep Ensemble (sin MI): {100 * ratio.mean():.1f}% (rango {100 * ratio.min():.1f}% a {100 * ratio.max():.1f}%)')
""")

md("""
## 4. Validación vs test

F1 promedio en validación (los datos con que se eligió todo) y en test. Si un método gana mucho en validación y nada en test, lo que ganó es sobreajuste.
""")

code("""
gap = scores.groupby('método', sort=False)[['F1 val (%)', 'F1 (%)', 'umbral']].mean()
gap = gap.rename(columns={'F1 (%)': 'F1 test (%)', 'umbral': 'umbral medio'})
gap['val − test'] = gap['F1 val (%)'] - gap['F1 test (%)']
display(gap.round(3))
""")

md("""
## Hallazgos

Promedio de los 15 folds de test.

1. **Mezclar ayuda muy poco, y con las dos formas da casi lo mismo.** El Deep Ensemble solo da F1 83.57. La grilla elegida en validación da 83.61 (+0.04, mejora en 7/15 folds) y la logística 83.56 (−0.01, 6/15). Ninguna diferencia es significativa (p > 0.5).
2. **La grilla muestra una meseta.** Cualquier peso de Neural Process entre 0.1 y 0.5 sube el F1 de test entre +0.05 y +0.13 (mejora en 9 a 11 de 15 folds). En validación esa meseta es todavía más plana (83.79–83.86), así que la validación **no logra elegir bien el peso**: en algunos folds elige 0 y en otros 0.8. Por eso el peso "elegido" rinde menos que un 50/50 fijo.
3. **La logística confía casi solo en el Deep Ensemble.** Sin MI, el logit de Neural Process recibe ~8% del peso del Deep Ensemble (coeficientes 0.06 contra 0.85; entre −3% y 27% según el fold). Es la misma conclusión del notebook 09, ahora a nivel de timestep: Neural Process aporta poca información nueva.
4. **Con MI, el modelo aprende algo razonable, pero no le sirve al F1.** Le baja el peso al Deep Ensemble cuando su MI es alta (coeficiente *logit p_DE × log MI_DE* negativo en los 15 folds) y, en casi todos los folds, le da más peso a Neural Process cuando la MI de NP es alta. Aun así, el F1 queda en 83.50.
5. **Lo que sí mejora la logística es la calibración.** ECE 0.0026–0.0027 contra 0.0037 del Deep Ensemble (~30% menos) y NLL 0.0468 contra 0.0473. La probabilidad mezclada es la mejor calibrada del proyecto, aunque no detecte más husos.
6. **Elegir también el umbral no ayuda.** Grilla + umbral gana +0.55 en validación y pierde −0.04 en test: sobreajuste.

**Conclusión.** La mezcla de probabilidades no mejora de forma confiable la detección: la ganancia posible (~+0.1 de F1) es del tamaño del ruido entre folds. Si se quiere una sola probabilidad para reportar, la **logística p_DE + p_NP** es la mejor opción por calibración. Para el F1, el Deep Ensemble solo sigue siendo la referencia más simple. Si en el informe se usa un 50/50 fijo, hay que decir que se miró en test y no se eligió en validación.
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
