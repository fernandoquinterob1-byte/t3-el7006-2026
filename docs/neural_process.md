# Neural Process sobre SEED

El modelo nuevo usa la implementación PyTorch de [Neural Process
Family](https://github.com/YannDubs/Neural-Process-Family) desde
`third_party/Neural-Process-Family`. El checkpoint de SEED se carga como
backbone congelado y el Neural Process aprende un adaptador probabilístico
binario sobre sus características y probabilidades.

## Dependencia fuente

Si se prepara un checkout nuevo:

```powershell
New-Item -ItemType Directory -Force third_party | Out-Null
git clone --depth 1 --filter=blob:none --sparse https://github.com/YannDubs/Neural-Process-Family.git third_party/Neural-Process-Family
git -C third_party/Neural-Process-Family sparse-checkout set npf
```

La librería no se instala como un paquete PyPI convencional; el código del
proyecto añade esa raíz a `sys.path` mediante `src/eeg_uq/npf_compat.py`.

## Entrenamiento de un fold

Debe existir el checkpoint SEED correspondiente, por ejemplo:

```text
outputs/seed_batch8_earlystop/train_eval_0_fold_0_seed_42/checkpoints/best_full.pt
```

Para una prueba diagnóstica:

```powershell
.\.venv\Scripts\python.exe train_neural_process.py `
  --evaluation 0 --fold 0 `
  --data-root "C:\ruta\data" `
  --seed-run-root outputs\seed_batch8_earlystop `
  --output-root outputs\neural_process `
  --debug --no-export-predictions
```

Para entrenar y exportar validation/test:

```powershell
.\.venv\Scripts\python.exe train_neural_process.py `
  --evaluation 0 --fold 0 `
  --data-root "C:\ruta\data" `
  --seed-run-root outputs\seed_batch8_earlystop `
  --output-root outputs\neural_process
```

Para varios folds:

```powershell
.\.venv\Scripts\python.exe run_np_cv.py `
  --evaluations 0 --folds 0 1 2 3 4 `
  --data-root "C:\ruta\data" `
  --seed-run-root outputs\seed_batch8_earlystop
```

## Protocolo

SEED produce 500 posiciones internas por ventana. El Neural Process recibe
esas características junto con tiempo normalizado y el logit SEED. El
contexto usa probabilidades SEED, por lo que no requiere etiquetas durante
validation/test. Las predicciones se repiten por 8 para conservar el
stitching y el postprocesamiento de eventos existentes.

`validation_predictions.npz` queda disponible para calibración Conformal y
`test_predictions.npz` permanece separado. Los resultados guardan también el
hash del checkpoint SEED fuente y del checkpoint del Neural Process.

