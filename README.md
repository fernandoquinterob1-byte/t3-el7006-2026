# Incertidumbre en EEG con SEED

El proyecto conserva el baseline SEED original y añade análisis de
incertidumbre post-hoc. La guía práctica única está en
[`docs/workflow.md`](docs/workflow.md).

## Regla principal

`eeg_code/` es el baseline legacy y no se modifica. La arquitectura de SEED,
preprocesamiento, folds, stitching, postprocesamiento, IoU y F1 se preservan.
El código reutilizable nuevo vive en `src/eeg_uq/`; las herramientas de
Conformal viven en `conformal/` y Laplace Redux en `laplace/`.

```text
EEG → SEED entrenado → probabilidades temporales → Conformal Prediction
```

Conformal no entrena una red. Usa probabilidades del mejor checkpoint sobre
validation para calibrar y sobre test para evaluar. Test nunca se usa para
calibrar.

Laplace Redux usa una copia del checkpoint para estimar incertidumbre de la
última capa y no modifica los pesos guardados de SEED. La guía está en
[`docs/laplace_redux.md`](docs/laplace_redux.md).

La integración de Neural Process sobre el checkpoint SEED está documentada en
[`docs/neural_process.md`](docs/neural_process.md).

## Dónde mirar

- [`docs/workflow.md`](docs/workflow.md): comandos de entrenamiento, export e
  Conformal; configuración y outputs.
- [`docs/conformal_prediction.md`](docs/conformal_prediction.md): método,
  fórmulas, métricas y limitaciones a nivel evento.
- [`docs/seed_baseline_reference.md`](docs/seed_baseline_reference.md):
  protocolo de referencia del baseline.
- [`notebooks/02_eval0_fold_dashboard.ipynb`](notebooks/02_eval0_fold_dashboard.ipynb):
  dashboard que lee los resultados de los cinco folds de evaluación 0.

Los datos privados permanecen fuera del repositorio. La carpeta indicada por
`--data-root` debe contener `moda_augmented_s5.h5` (entrenamiento) y
`moda_moda.h5` (inferencia raw), junto con los pesos opcionales del baseline.

## Estructura

```text
configs/default.yaml  única configuración activa del experimento
src/eeg_uq/           modelo, entrenamiento, inferencia y Conformal reutilizable
eeg_code/             baseline legacy aislado
conformal/            CLI y gráficos de Conformal
outputs/baseline/     resultados SEED legacy
outputs/conformal/    resultados Conformal
```
