"""Write notebooks/12_histeresis.ipynb (cells only; execute with scripts/run_notebook.py)."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "12_histeresis.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Postprocesado con histéresis y duración mínima

**Problema.** El postprocesado original decide qué es un huso así: probabilidad ≥ 0.425 durante **al menos 0.3 s seguidos** (además une huecos cortos y recorta eventos de más de 5 s). Unos 307 husos reales (~11% de los FN del Deep Ensemble) tienen probabilidad sobre 0.425, pero por menos de 0.3 s, y esa regla los borra.

**Idea.** Cambiar solo la regla final, sin tocar ninguna red:

- **Histéresis (dos umbrales)**: un evento *nace* si la probabilidad supera un umbral **alto** y *se extiende* mientras esté sobre un umbral **bajo**. Un pico corto pero claro se alarga y sobrevive al mínimo de 0.3 s.
- **Duración mínima**: el mismo umbral 0.425, con otro mínimo (0.2 a 0.4 s).
- **Las dos juntas** (primero el mejor par de umbrales, después la mejor duración mínima para ese par).

**Protocolo.** El código original (`eeg_code/`, `metrics.process_spindle_simple`) no se modifica: la variante está en `src/eeg_uq/hysteresis.py`, y con umbral alto = bajo = 0.425 y mínimo 0.3 s reproduce exactamente el resultado original (está probado en `tests/test_hysteresis.py`). En cada fold, los parámetros se eligen con la F1 de **validación** y se miden **una vez** en **test**. Se prueba sobre Deep Ensemble, SEED y Neural Process; 15 folds.
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
from eeg_uq import hysteresis as hy
from eeg_uq import report
from eeg_uq.hybrid_threshold import event_scores
from eeg_uq.method_summary import load_fold

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 250)


def show(figure):
    display(figure)
    plt.close(figure)

REFRESH = False  # True recalcula desde los NPZ (~10 minutos)
results = hy.run_all(refresh=REFRESH)
scores, grid = results['scores'], results['grid']
""")

md("""
## 1. Resultado en test

Δ es la diferencia **pareada** por fold contra el postprocesado **original** del mismo método.
""")

code("""
display(hy.summary(scores))
""")

md("""
## 2. Parámetros elegidos en validación

Mediana entre folds. La histéresis elegida (umbral alto ~0.5, bajo ~0.35) **sube** el umbral para empezar un evento: termina siendo un filtro de FP, no un rescate de FN.
""")

code("""
display(scores.groupby(['método', 'variante'], sort=False)[['umbral alto', 'umbral bajo', 'duración mínima (s)', 'F1 val (%)', 'F1 (%)']]
        .median().round(3).rename(columns={'F1 (%)': 'F1 test (%)'}))
""")

md("""
### Grilla de la histéresis en validación (Deep Ensemble)

F1 de validación promedio de los 15 folds para cada par (umbral alto, umbral bajo). La celda (0.425, 0.425) es el postprocesado original.
""")

code("""
de_grid = grid[(grid['método'] == 'Deep Ensemble') & (grid['variante'] == 'Histéresis')]
heat = de_grid.groupby(['umbral bajo', 'umbral alto'])['F1 val (%)'].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
image = ax.imshow(heat.to_numpy(), cmap='viridis', origin='lower', aspect='auto')
ax.set_xticks(range(heat.shape[1]), heat.columns)
ax.set_yticks(range(heat.shape[0]), heat.index)
ax.set_xlabel('umbral alto (para empezar un evento)')
ax.set_ylabel('umbral bajo (para continuarlo)')
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        if not np.isnan(heat.iat[i, j]):
            ax.text(j, i, f'{heat.iat[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white')
fig.colorbar(image, label='F1 validación (%)')
ax.set_title('Deep Ensemble: F1 de validación por par de umbrales')
show(fig)
""")

md("""
## 3. ¿Por qué no recupera los husos cortos?

Diagnóstico con **ajustes fijos** (no elegidos con ningún dato, solo para entender): manteniendo el umbral alto en 0.425 y bajando el umbral bajo o la duración mínima. Se cuenta cuántos husos reales se recuperan (Δ TP) y cuántas falsas alarmas entran (Δ FP), por fold de test.
""")

code("""
settings = {'original (0.425 / 0.425 / 0.30 s)': (0.425, 0.425, 0.30), 'umbral bajo 0.35': (0.425, 0.35, 0.30),
            'umbral bajo 0.30': (0.425, 0.30, 0.30), 'umbral bajo 0.25': (0.425, 0.25, 0.30),
            'duración mínima 0.20 s': (0.425, 0.425, 0.20), 'duración mínima 0.15 s': (0.425, 0.425, 0.15)}
rows = []
for evaluation in report.EVALUATIONS:
    for fold in report.FOLDS:
        data = load_fold(report.method_path('Deep Ensemble', evaluation, fold), 'sampling')
        for name, params in settings.items():
            rows.append({'ajuste': name, **event_scores(data['y_true'], hy.detect(data['probability'], *params))})
fixed = pd.DataFrame(rows).groupby('ajuste', sort=False).mean()
fixed['Δ TP (husos recuperados)'] = fixed['TP'] - fixed['TP'].iloc[0]
fixed['Δ FP (falsas alarmas nuevas)'] = fixed['FP'] - fixed['FP'].iloc[0]
display(fixed.round(2))
""")

md("""
## Hallazgos

Promedio de los 15 folds de test.

1. **Ninguna variante mejora el F1.** En el Deep Ensemble: histéresis −0.05, duración mínima +0.01, las dos juntas −0.07 (ninguna significativa). En SEED y Neural Process, entre −0.27 y −0.04.
2. **La histéresis elegida en validación termina filtrando FP, no rescatando FN.** Usa umbral alto ~0.5 y bajo ~0.35: exige más para empezar un evento y sube la precisión (+0.7 en el Deep Ensemble) a cambio de recall (−0.7). Es otra forma de mover el punto de operación, como las reglas del notebook 09.
3. **Rescatar los husos cortos sí funciona, pero trae ruido.** Con umbral bajo 0.35 se recuperan ~16 husos reales por fold, pero los mismos picos cortos aparecen también donde no hay huso, y entran ~23 falsas alarmas. Bajar la duración mínima a 0.2 s da lo mismo: +11 husos, +20 FP. El F1 queda igual (83.58 vs 83.57).
4. **El postprocesado original ya está bien ajustado.** Su mínimo de 0.3 s está justo donde recuperar un huso corto cuesta lo mismo que una falsa alarma nueva.

**Conclusión.** Los husos cortos que se pierden no se distinguen, por su probabilidad, de los picos cortos de ruido. Cambiar la regla final solo cambia un error por otro, igual que las reglas con incertidumbre. Para recuperar husos hace falta que el **modelo** los vea mejor (reentrenar o agregar diversidad), no un postprocesado distinto.
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
