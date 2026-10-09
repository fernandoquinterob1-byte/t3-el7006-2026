# Temporal Ensemble por fase sobre SEED

El Temporal Ensemble pasa la misma señal por SEED varias veces, desplazando el
corte de las ventanas, y mide cuánto cambian las predicciones. No entrena nada:
usa el checkpoint `best_full.pt` de cada fold sin modificarlo.

## Por qué por fase y no por segundos

Un diagnóstico sobre el test del fold 0 de la evaluación 0 midió cuánto cambia
la probabilidad de un instante según dónde cae dentro de la ventana de 20 s:

| Posición en la ventana | \|Δp\| medio cerca de eventos | Cambia la decisión |
|---|---|---|
| 0–2 s (borde) | 0.033 | 3.7% |
| 4–16 s (zona central) | 0.0004–0.0014 | 0.04–0.16% |
| 18–20 s (borde) | 0.019 | 2.2% |

SEED solo es sensible en los bordes de la ventana, y la costura legacy descarta
esos bordes. Por eso, desplazar las ventanas por segundos y quedarse con la
zona central produce miembros casi idénticos (MI del orden de 10⁻⁵ bits): el
método no tiene nada que medir. Incluir los bordes sí genera variación, pero
mezcla predicciones que se sabe que son peores.

La fuente de variación útil es la **fase** de la reducción ×8 de la red: si el
registro se desplaza 1–7 muestras, cambia cómo se agrupan las muestras al
reducir la resolución, aunque cada instante siga en la zona central.

| Miembros (diagnóstico, fold 0) | MI media cerca de eventos | AUROC errores (std) | AUPRC errores (std) |
|---|---|---|---|
| 1 s, solo zona central | 3e-5 bits | 0.67 | 0.24 |
| 1 s, ventana completa | 8e-3 bits | 0.85 | 0.30 |
| Fase 0–7 muestras | 2e-3 bits | 0.87 | 0.34 |
| Referencia: SEED H(p) | — | 0.895 | 0.417 |

## Implementación

- `src/eeg_uq/uncertainty/temporal_ensemble.py`: cada miembro `d` aplica la
  costura legacy de 20 s / 10 s a `signal[d:]` y se realinea al tiempo absoluto.
  Las primeras `d` muestras del registro (milisegundos) se toman del miembro sin
  desplazamiento.
- `temporal_ensemble/run_temporal_ensemble.py`: corre validación y test de cada
  fold y guarda los NPZ con el mismo formato que Laplace
  (`predictive_probabilities` de forma miembros × registros × tiempo,
  `mean_probability`, entropías y `mutual_information`).
- La detección usa la probabilidad media de los miembros con los umbrales y el
  post-procesamiento legacy.

## Interpretación

La MI del Temporal Ensemble mide **sensibilidad al corte temporal**, no
incertidumbre epistémica sobre los pesos. Se presenta como estabilidad o
robustez del detector.

## Ejecución

```powershell
.\.venv\Scripts\python.exe temporal_ensemble\run_temporal_ensemble.py --data-root "C:\ruta\a\datos"
```

La sección `temporal_ensemble` de `configs/default.yaml` define los folds, los
desplazamientos de fase y la carpeta de salida (`outputs/temporal_ensemble/`).
