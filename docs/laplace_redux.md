# Laplace Redux post-hoc sobre SEED

Laplace se ejecuta después del entrenamiento. El checkpoint original
`best_full.pt` se carga en memoria, se congela y se envuelve en un adapter.
El Laplace se ajusta sobre la última capa real de SEED, `classifier.logits`
(lineal 128 → 1), que recibe las features `ReLU(classifier.hidden(·))`. Como
laplace-torch usa verosimilitud categórica, esa capa se reescribe con dos
logits, pesos `[0; w]` y sesgo `[0; b]`, de modo que en el punto MAP:

```text
softmax([0, z])[1] = sigmoid(z),   z = w·h + b
```

Por tanto, la predicción media inicial conserva la salida binaria de SEED y la
incertidumbre depende de las features `h` de cada timestep. Las features se
calculan una vez por registro con las mismas ventanas de 20 s / 10 s y costura
central de la inferencia de SEED; la probabilidad se repite ×8 al final, igual
que SEED. No se reentrena ni se sobrescribe SEED.

## Cuántos datos ve el Laplace

La incertidumbre de Laplace baja a medida que crece la cantidad de datos
independientes, así que esa cantidad tiene que ser honesta:

- se usan los registros crudos (`moda_moda.h5`) de los sujetos de train, no las
  copias aumentadas de `moda_augmented_s5.h5`;
- se usa un punto por cada salida de SEED (una cada 8 muestras), no las 8 copias
  de cada logit;
- los puntos vecinos son casi iguales, por lo que la verosimilitud se divide por
  el tiempo de autocorrelación τ del logit de SEED (`temperature: auto`): `N`
  puntos correlacionados cuentan como `N / τ` independientes.

`metrics.json` registra los puntos de train, τ y los puntos efectivos.

## Datos y separación

- Los sujetos de entrenamiento calculan la curvatura posterior de Laplace.
- Validation se usa solamente para elegir `prior_precision` (grid search que
  minimiza la NLL de validación).
- Test se mantiene separado y sólo se evalúa.

La salida queda separada del entrenamiento:

```text
outputs/laplace_redux/eval_1_fold_1/
  laplace_posterior.pt
  validation_predictions.npz
  test_predictions.npz
  metrics.json
```

## Métricas

Cada split registra, sobre timesteps válidos:

- entropía predictiva;
- Mutual Information/BALD;
- NLL;
- ECE;
- muestras de probabilidad, si `save_predictive_samples` está activo.

Conformal es un experimento independiente y debe ejecutarse desde `conformal/`
usando los artefactos de predicciones del modelo que se quiera comparar.

## Ejecución

Configurar `paths.data_root` o `EEG_DATA_ROOT` y revisar la sección `laplace`
de `configs/default.yaml`. Después:

```powershell
.\.venv\Scripts\python.exe laplace\run_laplace.py --config configs\default.yaml
```

Se pueden ejecutar otros folds con overrides opcionales, sin cambiar el YAML
ni el checkpoint:

```powershell
.\.venv\Scripts\python.exe laplace\run_laplace.py --evaluation 0 --fold 1 --data-root "C:\ruta\a\datos"
```
