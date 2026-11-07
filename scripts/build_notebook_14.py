"""Write notebooks/14_analisis_fn.ipynb (cells only; execute with scripts/run_notebook.py)."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "14_analisis_fn.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# ¿Cómo son los husos que el modelo no detecta?

**Pregunta.** Ningún experimento logró bajar los falsos negativos sin subir los falsos positivos. ¿Por qué? ¿Los husos perdidos son débiles, raros, de ciertos sujetos, o anotaciones dudosas?

**Grupos** (Deep Ensemble, 15 folds de test):

- **TP**: husos anotados que el modelo detectó.
- **FN visto pero corto**: el modelo superó 0.425 dentro del huso, pero por menos de 0.3 s y el postprocesado lo borró.
- **FN casi visto**: probabilidad máxima entre 0.25 y 0.425.
- **FN invisible**: probabilidad máxima < 0.25. El modelo no lo sospechó.
- **FP**: detecciones sin huso anotado (medidas en el tramo predicho).
- **Fondo**: ventanas de 0.5 s al azar, a más de 1 s de cualquier huso anotado o detectado (referencia de "no huso").

**Qué se mide en la señal** (canal C3, 200 Hz): duración; amplitud pico a pico y RMS en la **banda sigma** (11–16 Hz, la frecuencia de los husos); **contraste sigma** = RMS sigma dentro del evento / RMS sigma en los ±2 s alrededor; RMS de banda ancha. Además, la probabilidad y la incertidumbre de los modelos. Código: `src/eeg_uq/fn_analysis.py`; no se entrena nada.
""")

code("""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import roc_auc_score

_candidates = [Path.cwd(), *Path.cwd().parents]
REPO_ROOT = next((p for p in _candidates if (p / 'configs' / 'default.yaml').exists()), Path.cwd())
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))
from eeg_uq import fn_analysis as fa
from eeg_uq import report
from eeg_uq.method_summary import load_fold

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 250)


def show(figure):
    display(figure)
    plt.close(figure)

REFRESH = False  # True recalcula desde los NPZ y el HDF5 (~2 minutos)
events = fa.run_all(refresh=REFRESH)
COLORS = {'TP': '#2a78d6', 'FN visto pero corto': '#f4a3a3', 'FN casi visto': '#e87070', 'FN invisible': '#c0392b',
          'FP': '#eb6834', 'Fondo': '#a0a0a0'}
""")

md("""
## 1. ¿Cuántos FN hay de cada tipo?
""")

code("""
counts = events['grupo'].value_counts().reindex(fa.GROUPS)
fn_total = counts[['FN visto pero corto', 'FN casi visto', 'FN invisible']].sum()
table = pd.DataFrame({'eventos (15 folds)': counts, 'por fold': (counts / report.N_FOLDS_TOTAL).round(1)})
table['% de los FN'] = (100 * counts / fn_total).where(counts.index.str.startswith('FN')).round(1)
display(table)
""")

md("""
## 2. ¿Cómo son en la señal?

Medianas por grupo. Debajo, gráficos de cajón de la amplitud sigma y del contraste sigma (sin valores extremos).
""")

code("""
display(events.groupby('grupo')[fa.FEATURES].median().reindex(fa.GROUPS).T.round(3))
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
for ax, feature in zip(axes, ['amplitud sigma pico a pico (µV)', 'contraste sigma (veces el fondo)', 'duración (s)']):
    values = [events.loc[events['grupo'] == g, feature].dropna() for g in fa.GROUPS]
    box = ax.boxplot(values, tick_labels=[g.replace('FN ', 'FN\\n') for g in fa.GROUPS], patch_artist=True, showfliers=False,
                     medianprops={'color': 'black'})
    for patch, group in zip(box['boxes'], fa.GROUPS):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.75)
    ax.set_title(feature)
    ax.tick_params(axis='x', labelsize=8)
show(fig)
""")

md("""
### ¿Cuánto se parecen a un TP o al fondo?

*Parecido al fondo*: amplitud **y** contraste sigma por debajo del percentil 75 del fondo. *Parecido a un TP típico*: amplitud **y** contraste por encima del percentil 25 de los TP. AUROC: qué tan bien cada medida separa ese grupo de los TP (0.5 = no los separa).
""")

code("""
amp, con = 'amplitud sigma pico a pico (µV)', 'contraste sigma (veces el fondo)'
tp, bg = events[events['grupo'] == 'TP'], events[events['grupo'] == 'Fondo']
rows = []
for group in fa.GROUPS:
    part = events[events['grupo'] == group]
    row = {'grupo': group,
           '% parecido al fondo': 100 * ((part[amp] <= bg[amp].quantile(.75)) & (part[con] <= bg[con].quantile(.75))).mean(),
           '% parecido a un TP típico': 100 * ((part[amp] >= tp[amp].quantile(.25)) & (part[con] >= tp[con].quantile(.25))).mean()}
    if group != 'TP':
        both = pd.concat([tp, part])
        target = (both['grupo'] == group).to_numpy()
        for feature in ('duración (s)', amp, 'RMS sigma (µV)', con):
            row[f'AUROC vs TP: {feature}'] = roc_auc_score(target, -both[feature])  # lower = this group
    rows.append(row)
display(pd.DataFrame(rows).set_index('grupo').round(3))
""")

md("""
## 3. ¿Qué tan seguro está el modelo cuando los pierde?

Medianas de probabilidad e incertidumbre. Un FN con incertidumbre **baja** es un error *silencioso*: el modelo está seguro de que no hay huso.
""")

code("""
display(events.groupby('grupo')[['p máx DE', 'p media DE', 'aleatoria media DE', 'MI media DE', 'MI media NP']]
        .median().reindex(fa.GROUPS).round(3))
""")

md("""
## 4. ¿Se concentran en algunos sujetos?
""")

code("""
annotated = events[~events['grupo'].isin(['FP', 'Fondo'])]
by_subject = annotated.groupby('sujeto')['grupo'].agg(husos='size', invisibles=lambda g: (g == 'FN invisible').sum())
by_subject['tasa de FN invisible'] = by_subject['invisibles'] / by_subject['husos']
ordered = by_subject.sort_values('invisibles', ascending=False)
top = int(np.ceil(0.1 * len(ordered)))
print(f'Sujetos con husos anotados: {len(ordered)}; con al menos un FN invisible: {(ordered["invisibles"] > 0).sum()}')
print(f'El 10% de sujetos con más FN invisibles ({top}) reúne {100 * ordered["invisibles"].iloc[:top].sum() / ordered["invisibles"].sum():.1f}% de ellos')
display(by_subject['tasa de FN invisible'].describe(percentiles=[.25, .5, .75, .9]).round(3).to_frame().T)
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
ax.scatter(by_subject['husos'], by_subject['tasa de FN invisible'], s=14, color='#c0392b', alpha=0.6)
ax.set_xlabel('husos anotados del sujeto')
ax.set_ylabel('fracción de FN invisibles')
ax.set_title('FN invisibles por sujeto')
show(fig)
""")

md("""
## 5. ¿Algo los distingue donde el modelo duda?

Los tramos *casi detectados* del notebook 09 (probabilidad ≥ 0.25 sin llegar a detección) mezclan husos reales y falsos. Si la amplitud sigma separara los reales, se podría usar para rescatarlos. AUROC en test (15 folds), comparado con la probabilidad y la MI de Neural Process; en las detecciones, para encontrar los FP.
""")

code("""
separability = fa.near_miss_separability(refresh=REFRESH)
for group in separability['grupo'].unique():
    print(group)
    display(report.by_evaluation(separability[separability['grupo'] == group],
                                 [c for c in separability.columns if c.startswith(('AUROC', 'azar'))]))
""")

md("""
## 6. Ejemplos

Evaluación 0, fold 0: tres **FN invisibles** (cerca de la mediana de amplitud sigma de su grupo), un **TP** y dos **FP** que se parecen a un TP típico. Paneles: señal EEG (µV), banda sigma, probabilidad del Deep Ensemble con el umbral 0.425. Verde = huso anotado; naranja punteado = el evento del ejemplo.
""")

code("""
fold_events = events[(events['evaluación'] == 0) & (events['fold'] == 0)]
fold = load_fold(report.method_path('Deep Ensemble', 0, 0), 'sampling')
row_of = {int(s): r for r, s in enumerate(fold['sample_id'])}


def pick(group, n, condition=None):
    part = fold_events[fold_events['grupo'] == group]
    if condition is not None:
        part = part[condition(part)]
    median = part[amp].median()
    return part.iloc[(part[amp] - median).abs().argsort()[:n]]

examples = pd.concat([pick('FN invisible', 3), pick('TP', 1),
                      pick('FP', 2, lambda p: (p[amp] >= tp[amp].quantile(.25)) & (p[con] >= tp[con].quantile(.25)))])
sr = 200
fig, axes = plt.subplots(3, len(examples), figsize=(3.4 * len(examples), 6.4), constrained_layout=True, sharey='row')
for col, (_, event) in enumerate(examples.iterrows()):
    raw, sig = fa.load_example(int(event['registro']))
    center = (int(event['inicio']) + int(event['fin'])) // 2
    lo, hi = max(0, center - 3 * sr), min(len(raw), center + 3 * sr)
    t = np.arange(lo, hi) / sr
    row = row_of[int(event['registro'])]
    truth = fold['y_true'][row, lo:hi]
    for ax, values in zip(axes[:, col], (raw[lo:hi], sig[lo:hi], fold['probability'][row, lo:hi])):
        ax.plot(t, values, color='#333333', linewidth=0.7)
        ax.fill_between(t, 0, 1, where=truth > 0, transform=ax.get_xaxis_transform(), color='#2ca25f', alpha=0.2)
        for x in (event['inicio'], event['fin']):
            ax.axvline(x / sr, color='#eb6834', linestyle=':', linewidth=1)
    axes[2, col].axhline(0.425, color='#c0392b', linestyle='--', linewidth=0.8)
    axes[2, col].set_ylim(0, 1)
    axes[0, col].set_title(f"{event['grupo']}\\nsigma {event[amp]:.0f} µV, contraste {event[con]:.1f}", fontsize=9)
    axes[2, col].set_xlabel('s')
axes[0, 0].set_ylabel('EEG (µV)')
axes[1, 0].set_ylabel('sigma (µV)')
axes[2, 0].set_ylabel('p Deep Ensemble')
show(fig)
""")

md("""
## Hallazgos

Deep Ensemble, 15 folds de test.

1. **Casi la mitad de los FN son invisibles.** De 2 727 husos perdidos, 1 322 (48%) tienen probabilidad máxima < 0.25; 889 (33%) son *casi vistos* y 516 (19%) se vieron por menos de 0.3 s.
2. **Los FN invisibles son husos débiles, pero reales.** Comparados con los TP son más cortos (0.51 vs 0.79 s), tienen ~37% menos amplitud sigma (20.8 vs 33.1 µV) y menos contraste con el fondo (1.8 vs 2.7 veces). La amplitud sigma los separa bien de los TP (AUROC ~0.85). Pero **no son fondo**: tienen 2.3 veces la amplitud sigma del fondo, y solo ~4% se parece al fondo. Es decir, hay actividad de huso, solo que débil.
3. **Son errores silenciosos.** En los FN invisibles, la incertidumbre del Deep Ensemble es tan baja como en los aciertos (aleatoria 0.39 vs 0.38; MI 0.02 vs 0.015). El modelo está **seguro** de que no hay huso. Por eso ninguna regla basada en la incertidumbre puede encontrarlos.
4. **Donde el modelo duda, ni la señal distingue los husos reales.** En los *casi detectados*, la amplitud y el contraste sigma separan los husos anotados de los no anotados tan mal como la MI (AUROC 0.48–0.54). En ese rango, lo que el experto marcó y lo que no marcó se ve igual en la señal. Eso apunta a **ambigüedad en la anotación** de los husos débiles: es el techo del problema, no una falla que una regla pueda corregir.
5. **Un tercio de los FP se parece a un huso típico.** El 33% de las falsas alarmas tiene amplitud y contraste sigma de un TP típico. Algunas podrían ser husos reales no anotados: parte de los "FP" sería ruido de la etiqueta.
6. **Están repartidos entre sujetos.** 150 de 179 sujetos tienen al menos un FN invisible; el 10% de sujetos con más FN invisibles reúne ~36% de ellos. Hay sujetos más difíciles, pero el problema no es de unos pocos.

**Conclusión.** Los FN que quedan son en su mayoría **husos débiles y cortos**, cerca del límite de lo que un experto marca. El modelo los descarta con seguridad, y en la zona dudosa la señal no distingue lo anotado de lo no anotado. Esto explica por qué ningún umbral, mezcla o postprocesado mejoró el F1: el límite está en la **ambigüedad de los husos débiles**, no en cómo se usa la salida del modelo. Para avanzar haría falta más información de la anotación (por ejemplo, el acuerdo entre expertos por huso) o un modelo entrenado para ser más sensible a husos de baja amplitud, aceptando más FP.

*Cuidados*: las medidas de señal son simples (filtro de banda sigma, sin limpiar artefactos) y el fondo se muestrea lejos de los eventos, por lo que es un "no huso" fácil. Los ejemplos se eligen cerca de la mediana de su grupo, no al azar.
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
