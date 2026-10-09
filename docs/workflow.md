# Flujo operativo único

Este documento reúne los comandos habituales. Los parámetros compartidos viven
solamente en [`configs/default.yaml`](../configs/default.yaml): batch size 8,
50 épocas como máximo, paciencia 5 y los valores por defecto de Conformal.
Cambiar ese archivo cambia los valores futuros; un argumento de consola sólo
sobrescribe esa corrida.

## Entrenar SEED

Los índices de consola empiezan en cero. Por tanto, evaluación 0 y fold 0 son
`eval_1/fold_1` en el archivo de folds legacy.

```powershell
.\.venv\Scripts\python.exe run_cv.py --evaluations 0 --folds 0 1 2 3 4 --data-root "C:\Users\ferna\Downloads\data\data" --output-root "outputs\seed_batch8_earlystop"
```

Cada fold entrena SEED una vez, guarda `checkpoints/best_full.pt` y luego hace
dos pasadas de inferencia con ese mismo checkpoint: una sobre validation y otra
sobre test. No entrena una red adicional. Los archivos quedan juntos:

```text
outputs/seed_batch8_earlystop/train_eval_0_fold_0_seed_42/
  checkpoints/best_full.pt
  metrics.json
  validation_predictions.npz
  validation_metrics.json
  test_predictions.npz
  test_metrics.json
```

`validation_predictions.npz` se usa para calibrar Conformal. `test_predictions.npz`
queda separado y no participa en la calibración. Los modos `--debug` y
`--overfit-small-subset` no producen estos archivos porque no son estimaciones
experimentales válidas.

## Completar exports de una corrida anterior

Las cinco corridas que existían antes de esta mejora no tienen los dos NPZ raw.
Se pueden completar sin reentrenar; sólo se carga `best_full.pt` y se hace
inferencia:

```powershell
.\.venv\Scripts\python.exe export_test_predictions.py --evaluation 0 --fold 0 --run-dir "outputs\seed_batch8_earlystop\train_eval_0_fold_0_seed_42" --data-root "C:\Users\ferna\Downloads\data\data" --splits validation test --device cuda
```

Repetir cambiando `--fold` y `--run-dir` para los folds 1 a 4. Esta operación
no cambia pesos, arquitectura, preprocesamiento, thresholds, IoU ni F1 legacy.

## Ejecutar Conformal Prediction

Una vez que estén los dos NPZ del mismo fold:

```powershell
.\.venv\Scripts\python.exe conformal\run_conformal.py --calibration-predictions "outputs\seed_batch8_earlystop\train_eval_0_fold_0_seed_42\validation_predictions.npz" --test-predictions "outputs\seed_batch8_earlystop\train_eval_0_fold_0_seed_42\test_predictions.npz"
```

El valor por defecto es `alpha: 0.05`; el mismo archivo de configuración define
el barrido `0.01, 0.05, 0.10, 0.20`. Para una corrida concreta:

```powershell
.\.venv\Scripts\python.exe conformal\run_conformal.py --calibration-predictions "...\validation_predictions.npz" --test-predictions "...\test_predictions.npz" --alpha 0.05 --alphas 0.01 0.05 0.10 0.20
```

Los resultados se organizan en `outputs/conformal/eval_1_fold_1/`, con
`conformal_results.npz`, `metrics.json` y `plots/` en una sola carpeta.

## Separación de responsabilidades

```text
eeg_code/       baseline legacy conservado, sin cambios
src/eeg_uq/     lógica reutilizable: SEED, inferencia y Conformal
conformal/      CLI y plots específicos de Conformal
outputs/baseline/   resultados legacy
outputs/conformal/  resultados Conformal
```

Para la explicación estadística de Conformal y sus límites a nivel evento,
consultar [`conformal_prediction.md`](conformal_prediction.md). Para las
decisiones de referencia del baseline, consultar
[`seed_baseline_reference.md`](seed_baseline_reference.md).

## Ejecutar Laplace Redux

Laplace es una etapa post-hoc que carga `best_full.pt`, usa los sujetos de
entrenamiento para estimar la curvatura de la última capa y escribe resultados
en `outputs/laplace_redux/`. El checkpoint de SEED no se sobrescribe.

Los valores operativos están en la sección `laplace` de
[`configs/default.yaml`](../configs/default.yaml), incluyendo evaluación/fold,
checkpoint, número de muestras Monte Carlo, Hessian, prior y dispositivo:

```powershell
.\.venv\Scripts\python.exe laplace\run_laplace.py --config configs\default.yaml
```

El comando produce por fold `laplace_posterior.pt`, predicciones de validation
y test, y `metrics.json`. `metrics.json` incluye únicamente las métricas
probabilísticas de Laplace: entropía predictiva, Mutual Information/BALD,
NLL y ECE. Conformal se ejecuta aparte con `conformal/run_conformal.py` para
comparar cada modelo de forma independiente.
