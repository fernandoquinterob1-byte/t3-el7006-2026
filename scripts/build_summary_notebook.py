"""Write notebooks/11_resumen_experimentos.ipynb (cells only, no outputs).

The notebook reads ``eeg_uq.experiments_summary.EXPERIMENTS``, so a new
experiment only needs a registry entry there; this script only has to be
rerun when the static text (general conclusions) changes.  Execute the
notebook afterwards (VS Code, or ``jupyter nbconvert --execute``).
"""

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "11_resumen_experimentos.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Resumen de experimentos

Todos los experimentos del proyecto en un solo lugar, y al final una comparación entre ellos.

**Cómo leerlo**

- Cada sección es un experimento: la pregunta, qué se hizo, el resultado y la conclusión. El detalle está en el notebook indicado.
- **Referencia**: el **Deep Ensemble con el umbral original (0.425)**, el mejor método base. Todos los Δ son la diferencia **pareada** por fold contra él (positivo = mejor).
- Todo es sobre **test**, 15 folds (evaluaciones 0, 1 y 2). En cada experimento, lo que se ajusta se elige con **validación** y se mide una sola vez en test.
- *Folds que mejoran*: en cuántos de los 15 folds el F1 sube. *p Wilcoxon*: si el cambio de F1 es distinto de cero (p < 0.05 = diferencia confiable).
- F1, precisión y recall son por evento (IoU ≥ 0.2), con el emparejamiento del código original.

Los datos salen de las cachés de cada experimento (`outputs/results_reports/analysis/`) y de los `test_predictions.npz`: este notebook no entrena ni recalcula nada. Para agregar un experimento: una entrada en `EXPERIMENTS` (`src/eeg_uq/experiments_summary.py`) y volver a ejecutar.
""")

code("""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

_candidates = [Path.cwd(), *Path.cwd().parents]
REPO_ROOT = next((p for p in _candidates if (p / 'configs' / 'default.yaml').exists()), Path.cwd())
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))
from eeg_uq import experiments_summary as es
from eeg_uq import report

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 60)


def show(figure):
    display(figure)
    plt.close(figure)


scores = es.all_scores()
table = es.compare(scores)
COLUMNS = ['F1 (%)', 'precisión (%)', 'recall (%)', 'FP', 'FN', 'Δ F1', 'Δ precisión', 'Δ recall', 'folds que mejoran F1', 'p Wilcoxon']


def pretty(part):
    part = part.set_index('variante')[COLUMNS].copy()
    part['folds que mejoran F1'] = part['folds que mejoran F1'].astype(int).astype(str) + '/15'
    return part.round({'F1 (%)': 2, 'precisión (%)': 2, 'recall (%)': 2, 'FP': 1, 'FN': 1, 'Δ F1': 2,
                       'Δ precisión': 2, 'Δ recall': 2, 'p Wilcoxon': 3})

print(f'{len(es.EXPERIMENTS)} experimentos, {table.shape[0]} variantes')
""")

md("""
## Experimentos
""")

code("""
for number, experiment in enumerate(es.EXPERIMENTS, start=1):
    display(Markdown(
        f'### {number}. {experiment.title}\\n\\n'
        f'*Notebook {experiment.notebook}*\\n\\n'
        f'**Pregunta.** {experiment.question}\\n\\n'
        f'**Qué se hizo.** {experiment.what}'
    ))
    if experiment.loader is None:
        display(Markdown('*Análisis sin variantes de F1: no entra en la tabla comparativa.*'))
    else:
        display(pretty(table[table['experimento'] == experiment.key]))
    display(Markdown(f'**Conclusión.** {experiment.conclusion}'))
""")

md("""
## Comparación de todos los experimentos

### Tabla completa
""")

code("""
everything = table.set_index(['experimento', 'variante'])[COLUMNS].copy()
everything['folds que mejoran F1'] = everything['folds que mejoran F1'].astype(int).astype(str) + '/15'
display(everything.round(2))
""")

md("""
### La mejor variante según qué importe

Para precisión y recall solo se consideran variantes que no pierden más de 0.3 puntos de F1 (si no, cualquier regla extrema "gana" sacrificando todo lo demás).
""")

code("""
others = table[~((table['experimento'] == es.REFERENCE[0]) & (table['variante'] == es.REFERENCE[1]))]
keep_f1 = others[others['Δ F1'] >= -0.3]
picks = {
    'Mejor F1': others.loc[others['F1 (%)'].idxmax()],
    'Mejor precisión (Δ F1 ≥ −0.3)': keep_f1.loc[keep_f1['precisión (%)'].idxmax()],
    'Mejor recall (Δ F1 ≥ −0.3)': keep_f1.loc[keep_f1['recall (%)'].idxmax()],
}
reference = table[(table['experimento'] == es.REFERENCE[0]) & (table['variante'] == es.REFERENCE[1])].iloc[0]
best = pd.DataFrame({'Referencia (Deep Ensemble)': reference, **picks}).T
numeric = ['F1 (%)', 'precisión (%)', 'recall (%)', 'Δ F1', 'Δ precisión', 'Δ recall', 'p Wilcoxon']
best[numeric] = best[numeric].astype(float).round(3)
best['folds que mejoran F1'] = best['folds que mejoran F1'].astype(int).astype(str) + '/15'
display(best[['experimento', 'variante', *numeric[:6], 'folds que mejoran F1', 'p Wilcoxon']])
""")

md("""
### Precisión vs recall

Cada punto es una variante (promedio de 15 folds). Las curvas grises son F1 constante: moverse **a lo largo** de una curva cambia un error por otro; solo cruzarla **hacia arriba a la derecha** mejora el F1. La estrella es la referencia.
""")

code("""
palette = ['#52514e', '#2a78d6', '#eda100', '#1baf7a', '#e87ba4', '#4a3aa7', '#eb6834', '#7fb3ec']
colors = {e.key: palette[i % len(palette)] for i, e in enumerate(es.EXPERIMENTS)}
fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
recall_grid = np.linspace(78, 88, 200)
for f1 in (82.5, 83.0, 83.5, 84.0, 84.5):
    precision = f1 * recall_grid / (2 * recall_grid - f1)
    ax.plot(recall_grid, precision, color='#cccccc', linewidth=0.8, zorder=0)
    label_precision = 81.6  # where each curve is labelled, inside the axes
    label_recall = f1 * label_precision / (2 * label_precision - f1)
    if 80.5 < label_recall < 88:
        ax.text(label_recall, label_precision, f'F1 {f1}', fontsize=7, color='#888888', ha='center', va='bottom')
for experiment, part in table.groupby('experimento', sort=False):
    ax.scatter(part['recall (%)'], part['precisión (%)'], s=40, color=colors[experiment], label=experiment, alpha=0.85)
ax.scatter([reference['recall (%)']], [reference['precisión (%)']], marker='*', s=320, color='black', label='Referencia: Deep Ensemble', zorder=5)
ax.set_xlim(80.5, 88)
ax.set_ylim(81.5, 87)
ax.set_xlabel('recall (%)')
ax.set_ylabel('precisión (%)')
ax.set_title('Todas las variantes: precisión vs recall (test, 15 folds)')
ax.legend(fontsize=8, loc='upper right')
show(fig)
""")

md("""
### Cambio de F1 contra la referencia

Barra = Δ F1 promedio; línea = intervalo de confianza del 95% entre folds. Si el intervalo cruza el cero, el cambio no es confiable.
""")

code("""
ordered = others.sort_values('Δ F1')
fig, ax = plt.subplots(figsize=(10, 0.32 * len(ordered) + 1.5), constrained_layout=True)
y = np.arange(len(ordered))
ax.barh(y, ordered['Δ F1'], xerr=ordered['IC95 Δ F1'], color=[colors[e] for e in ordered['experimento']],
        alpha=0.85, error_kw={'ecolor': '#333333', 'capsize': 2, 'linewidth': 0.8})
ax.set_yticks(y, [f"{v}  [{e}]" for e, v in zip(ordered['experimento'], ordered['variante'])], fontsize=7)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Δ F1 contra el Deep Ensemble (puntos)')
ax.set_title('Cambio de F1 de cada variante (test, 15 folds, IC 95%)')
show(fig)
""")

md("""
## Conclusiones generales

*(Se actualizan con cada experimento nuevo.)*

1. **El Deep Ensemble con el umbral original es lo mejor en F1** (83.57%). Ningún experimento lo supera de forma confiable: los cambios van de −0.3 a +0.04 puntos, y ninguna mejora es significativa.
2. **La incertidumbre no reduce los errores; los cambia de lugar.** Las reglas que suben el recall bajan la precisión en la misma medida, y al revés. En el gráfico de precisión vs recall, todas las variantes quedan sobre la misma curva de F1 ≈ 83.5.
3. **Sí sirve para elegir el punto de operación:**
   - **Más precisión:** *Solo FP: logística solo DE*, precisión 85.9% (+1.5), recall 81.6%.
   - **Más recall:** *Regla 2: umbral adaptativo* (MI de Neural Process), recall 84.5% (+1.5), precisión 82.8%.
   - **Equilibrio:** el Deep Ensemble solo.
4. **Neural Process aporta poca información nueva sobre el Deep Ensemble.** Sus MI están correlacionadas (ρ ≈ 0.73) y la mezcla logística le da ~8% del peso. Lo único que mejora la mezcla es la **calibración** (ECE 0.0026 vs 0.0037).
5. **Cambiar el postprocesado tampoco ayuda.** La histéresis y otra duración mínima recuperan husos cortos (~+16 por fold), pero traen casi la misma cantidad de picos cortos de ruido (~+23 FP). El postprocesado original ya está bien ajustado.
6. **Más semillas iguales casi no suman.** El Deep Ensemble ya está cerca de su techo: ~83.68 con 10 semillas y ~83.77 con infinitas, solo +0.2 sobre el actual.
7. **El límite está en los husos débiles y en la anotación.** La mitad de los FN tiene probabilidad < 0.25: son husos reales pero débiles y cortos, que el modelo descarta **con seguridad** (errores silenciosos: la incertidumbre no los marca). Donde el modelo duda, ni la amplitud sigma distingue lo que el experto anotó de lo que no (AUROC ~0.5), y un tercio de los FP se parece a un huso típico. Ninguna regla, mezcla ni postprocesado puede superar esa ambigüedad. Para avanzar haría falta más información de la anotación (acuerdo entre expertos por huso) o un modelo más sensible a husos de baja amplitud, aceptando más FP.
8. **El uso más fuerte de la incertidumbre es la revisión humana.** No sirve para decidir sola mejor que la probabilidad, pero ordena bien qué revisar: el 20% de detecciones más inciertas contiene ~60% de los FP.
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
