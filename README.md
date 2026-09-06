# Estimación de incertidumbre en redes neuronales aplicada a medicina (EEG)

## Estado del baseline y regla de reproducibilidad

La referencia metodológica es [Tapia-Rivas, Estévez y Cortes-Briones (2024)](https://doi.org/10.1038/s41598-023-50736-7) y la referencia práctica es [`eeg_code/run_model.ipynb`](eeg_code/run_model.ipynb). El nuevo código **no rediseña SEED**: conserva la arquitectura PyTorch, `BCEWithLogitsLoss`, selección por sujeto, stitching, postprocesamiento, pairing, IoU y F1 del notebook. Sólo añade configuración portable, exportación y análisis post-hoc.

Antes de comparar números, leer [`docs/seed_baseline_reference.md`](docs/seed_baseline_reference.md): el artículo y el notebook no son idénticos en todos sus detalles (por ejemplo, contexto de bordes y máximo de duración). Para reproducir el valor histórico de este repositorio, manda el notebook legacy.

La arquitectura concreta preservada está descrita en [`docs/seed_architecture.md`](docs/seed_architecture.md); el dominio médico y el protocolo de eventos, en [`docs/medical_domain.md`](docs/medical_domain.md).

## Dataset location

Los datos privados no se versionan ni deben copiarse dentro del repositorio. Cada colaborador configura una única carpeta externa:

```text
<EEG_DATA_ROOT>/
├── moda_augmented_s5.h5          # entrenamiento nuevo
├── moda_moda.h5                  # inferencia baseline legacy
├── model_weights/                # pesos entrenados localmente, opcional
└── model_weights_best_original/  # checkpoints legacy para reproducir baseline
```

Los folds sí son pequeños y versionados en `eeg_code/data/moda_folds.json`. Configura la raíz por comando o variable de entorno:

```bash
# Linux
export EEG_DATA_ROOT=/mnt/eeg_private_data

# Windows PowerShell
$env:EEG_DATA_ROOT = 'D:\eeg_private_data'

# Siempre se puede pasar explícitamente
python scripts/eda.py --data-root /ruta/al/eeg_private_data
```

Cada CLI valida archivos y rutas antes de abrir HDF5. No existen rutas personales hardcodeadas en el pipeline nuevo. El notebook legacy se conserva sin cambios y aún espera sus paths relativos; usar las CLIs nuevas para trabajo reproducible.

## Setup for collaborators

```bash
git clone https://github.com/fernandoquinterob1-byte/t3-el7006-2026.git
cd t3-el7006-2026
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env                  # editar sólo de forma local, si se desea
```

`requirements.txt` raíz es la lista portable del pipeline nuevo. El archivo histórico `eeg_code/requirements.txt` es un volcado del entorno del notebook y no es el instalador recomendado.

## First uncertainty experiment

Los índices de CLI son **cero-based**: `--evaluation 0 --fold 0` equivale a `eval_1/fold_1` del JSON legacy. Los IDs internos nunca se aplanan: hay 3 evaluaciones × 5 folds.

```bash
# 1. Auditoría y EDA (agregar --include-raw y --plot-examples 2 sólo si se desea)
python scripts/eda.py --data-root "$EEG_DATA_ROOT"

# 2. Smoke test del pipeline de entrenamiento, sin evaluación legacy completa
python train.py --evaluation 0 --fold 0 --debug --data-root "$EEG_DATA_ROOT"

# 3. Verificación de memorización/debug de N ventanas
python train.py --evaluation 0 --fold 0 --overfit-small-subset 64 --epochs 20 --data-root "$EEG_DATA_ROOT"

# 4. Entrenar un modelo nuevo de un fold; W&B es opcional
python train.py --evaluation 0 --fold 0 --data-root "$EEG_DATA_ROOT" --logger none
python train.py --evaluation 0 --fold 0 --data-root "$EEG_DATA_ROOT" --logger wandb

# 5. Reproducir un fold con checkpoints legacy y guardar logits/probabilidades/eventos
python evaluate_baseline.py --evaluation 0 --fold 0 --data-root "$EEG_DATA_ROOT" --device cuda

# 6. Reproducir el microaverage histórico de los 15 pares (trabajo pesado)
python evaluate_baseline.py --data-root "$EEG_DATA_ROOT" --device cuda

# 7. Entropía/confianza binaria post-hoc
python analyze_uncertainty.py --predictions outputs/baseline/predictions_legacy_eval_1_fold_1.npz
```

Para W&B, instalar dependencias, ejecutar `wandb login` sólo en el equipo que lo usará y pasar `--logger wandb`. Con `--logger none` no se importa ni conecta W&B.

`--overfit-small-subset N` es deliberadamente un diagnóstico: evalúa el mismo subconjunto de entrenamiento, por lo que sus métricas no son métricas de validación ni deben reportarse como resultado experimental.

### Convención de índices y outputs

El JSON, los checkpoints y el evaluador interno usan `eval_1..eval_3` y `fold_1..fold_5` (**uno-based**). Todas las CLIs públicas usan `--evaluation 0..2 --fold 0..4` (**cero-based**), por lo que `0/0` siempre selecciona internamente `eval_1/fold_1`.

- `train.py` escribe `outputs/train_eval_0_fold_0_seed_42/predictions.npz`: sus campos `evaluation` y `fold` son cero-based.
- `evaluate_baseline.py` escribe `outputs/baseline/predictions_legacy_eval_1_fold_1.npz`: sus campos `legacy_evaluation` y `legacy_fold` son uno-based e incluye los campos `cli_*_zero_based` equivalentes.

Los NPZ no usan arrays `object`. Para un conjunto de `N` filas HDF5 y longitud temporal `T`, los outputs principales tienen shape `(N, T)`: `y_true` (`int8`), `logits` (`float32`), `probabilities` (`float32`), `y_pred` (`int8`) y `processed_pred` (`int8`, sólo baseline). `sample_id` tiene shape `(N,)`; `subject_id`, shape `(N,)` Unicode. Los eventos se guardan como arrays planos paralelos `(E,)` (`event_*`), no como listas de Python.

Guardar logits y probabilidades duplica aproximadamente 8 bytes por timestep. Un fold de entrenamiento MODA aumentado de aproximadamente 2 988 ventanas × 4 000 timesteps ocupa cerca de 114 MiB sin compresión para estas cinco matrices; un fold baseline sin aumento es menor. `np.savez_compressed` ya aplica compresión sin pérdida, pero logits suelen comprimir poco. Para corridas masivas, la solución sin pérdida recomendada es conservar `logits` float32 y derivar `probabilities = sigmoid(logits)` al analizar; por ahora se guardan ambos por claridad y para cumplir el MVP.

La CV no se ejecuta sola: cuando corresponda, lanzar explícitamente `python run_cv.py --evaluations 0 1 2 --folds 0 1 2 3 4 --data-root "$EEG_DATA_ROOT" --extra --device cuda`.

## Estructura y colaboración Git

```text
configs/         defaults versionados; local.yaml está ignorado
docs/            teoría, dominio médico y contraste de referencia
src/eeg_uq/      data, modelo, folds, evaluación, entrenamiento e incertidumbre
scripts/         EDA explícita
tests/           pruebas sintéticas sin GPU
outputs/         resultados locales, ignorados
eeg_code/        notebook legacy y sus metadatos/diagramas de referencia
```

Se ignoran datasets HDF5, pesos, `outputs/`, W&B, logs, caches y configuraciones locales. No se ignoran código, documentación, folds ni configuraciones reproducibles. Flujo recomendado: actualizar `main`, crear `feature/nombre`, hacer cambios pequeños, `git add`, `git commit`, `git push -u origin feature/nombre` y abrir PR. Evitar editar simultáneamente notebooks; la lógica nueva debe vivir en `src/` y las CLIs.

**Uncertainty on EEG**

**Guías:** Sebastián Guzmán

## Descripción del proyecto

Los modelos de redes neuronales que trabajan con datos médicos enfrentan un desafío fundamental: la sobreconfianza en regiones fuera de la distribución de entrenamiento. Esto puede llevar a diagnósticos erróneos o a la omisión de patrones poco frecuentes, como eventos atípicos en los EEG de sueño y alteraciones de los husos de sueño asociadas a trastornos como la narcolepsia.

En un contexto clínico, donde cada error puede impactar directamente la salud del paciente, cuantificar la incertidumbre es crucial para que los médicos tomen decisiones informadas y prioricen los casos que requieran mayor atención.

El proyecto explora y compara técnicas de estimación de incertidumbre aplicadas a la detección de husos de sueño y complejos K en señales EEG:

- **Ensembles temporales de ventanas desfasadas:** combinan predicciones sobre distintas ventanas de una misma señal para medir la estabilidad frente a pequeños desplazamientos temporales.
- **Métodos bayesianos y aproximaciones de Laplace:** en particular, aproximaciones sobre las últimas capas de la red para incorporar incertidumbre paramétrica con un costo computacional moderado.
- **Conformal Prediction:** construye conjuntos de predicción con garantías estadísticas de cobertura e identifica ejemplos ambiguos o potencialmente fuera de distribución.
- **Neural Processes y Kernel Learning:** proporcionan estimaciones basadas en kernels para medir la variabilidad de la distribución posterior de segmentación.

Cada enfoque ofrece una forma distinta de evaluar la confiabilidad de las predicciones y de diseñar modelos más robustos para señales EEG [7].

## Preguntas de investigación

- ¿Qué técnicas identifican con mayor confiabilidad señales EEG de alta incertidumbre?
- ¿Es posible ajustar un umbral de rechazo que permita detecciones más seguras?
- ¿Puede separarse la incertidumbre epistemológica de la aleatoria?
- ¿Qué diferencias presentan Laplace, Conformal Prediction y los ensembles de ventanas temporales en la clasificación de EEG?
- ¿Cómo se correlacionan las medidas de incertidumbre con los errores reales del modelo en la detección de husos de sueño y complejos K?
- ¿Pueden las medidas de incertidumbre detectar eventos raros o patrones tempranos de trastornos del sueño o TDAH?

## Software, datos y tutoriales

- **Software disponible:** PyTorch y TensorFlow. El código incluido actualmente usa **PyTorch**; TensorFlow no aparece como dependencia de los notebooks revisados.
- **Dificultad:** Alta.
- **Base de datos y código protegido:** solicitar acceso al tutor.
- **Dependencias registradas:** [`eeg_code/requirements.txt`](eeg_code/requirements.txt). Es un volcado del entorno original e incluye rutas locales `file://`; conviene validarlo antes de usarlo como archivo de instalación en otro equipo.

Tutoriales de referencia:

- [UQ-Toolbox](https://github.com/lightning-uq-box/lightning-uq-box)
- [Neural Processes](https://yanndubs.github.io/Neural-Process-Family/)
- [Laplace-redux](https://aleximmer.com/Laplace/)
- [Conformal Prediction](https://github.com/aangelopoulos/conformal-prediction)

## Guía del código base

### Estructura

| Ruta | Propósito |
| --- | --- |
| [`eeg_code/run_model.ipynb`](eeg_code/run_model.ipynb) | Notebook principal: definición del modelo SEED, entrenamiento por folds e inferencia. |
| [`eeg_code/data/moda_folds.json`](eeg_code/data/moda_folds.json) | Particiones de sujetos para `eval_1`–`eval_3` y `fold_1`–`fold_5`. |
| [`eeg_code/SEED_diagram.png`](eeg_code/SEED_diagram.png) | Diagrama de la arquitectura completa y sus dimensiones. |
| [`eeg_code/Conv_MDB_Diagram.png`](eeg_code/Conv_MDB_Diagram.png) | Diagrama de las ramas convolucionales con dilataciones múltiples. |
| [`eeg_code/inference_results_moda_microaverage.json`](eeg_code/inference_results_moda_microaverage.json) | Resultado agregado de inferencia registrado en el repositorio. |
| [`uncertainty_tutorial.ipynb`](uncertainty_tutorial.ipynb) | Tutorial independiente sobre ensemble, MC Dropout y máxima probabilidad con datos sintéticos. |

En `eeg_code/`, el código ejecutable está dentro de notebooks; no hay scripts `.py` ni una CLI configurada. Los paths son relativos al directorio de ejecución. Para usar `run_model.ipynb`, trabajar desde `eeg_code/` o adaptar explícitamente sus paths.

### Datos esperados

El notebook abre archivos HDF5 con estos datasets:

- `signals`: señales de entrada.
- `labels`: etiquetas binarias de segmentación.
- `subject_ids`: identificadores de sujeto usados para filtrar cada partición.

Los archivos HDF5 no están incluidos en este checkout. Las rutas usadas por el notebook son:

- Entrenamiento: `eeg_code/data/moda_augmented_s5.h5`.
- Inferencia: `eeg_code/data/moda_moda.h5` (sin augmentación).
- Folds: `eeg_code/data/moda_folds.json`.

### Arquitectura SEED

`SEED_Model` combina:

1. `ConvEmbed_SEED`, con convoluciones 1D, BatchNorm, ReLU, pooling promedio y reducción temporal.
2. `StageMultiDilatedConvolutions`, con ramas de convolución dilatada y concatenación de sus salidas.
3. `BiLSTM`, una LSTM bidireccional de dos capas con dropout.
4. `RED_FC`, una capa oculta de 128 unidades y una salida logística binaria.

La configuración registrada usa `EMBED_DIM=64`, `FILTER_IN=64`, `BLOCK_DILATATION=2`, kernel convolucional 3, `BCEWithLogitsLoss` y `RESOL_FACTOR=8`. El clasificador produce logits temporales que luego se repiten ocho veces para recuperar la resolución de salida.

### Entrenamiento

El flujo de entrenamiento de `run_model.ipynb` es:

1. Cargar el HDF5 y seleccionar muestras por `subject_ids`.
2. Para cada evaluación y fold, definir `train_subjects` excluyendo los sujetos de validación y prueba.
3. Crear un modelo y un optimizador nuevos para cada fold.
4. Balancear los batches de entrenamiento según la mediana de positivos por muestra.
5. Entrenar con AdamW, clipping de gradiente y evaluación por pérdida de validación.
6. Guardar los mejores y los últimos pesos de `embedding`, `transformer` y `classifier`.

Parámetros registrados: 5 épocas, batch de 32, `lr=1e-4`, `weight_decay=1e-4`, 12 workers y early stopping después de seis épocas sin mejora. La configuración de entrenamiento establece `norm_data=False`; no asumir normalización adicional sin documentarla.

El archivo de folds contiene 3 evaluaciones × 5 folds. Cada fold lista 36 sujetos de validación y 36 de prueba; el conjunto de entrenamiento es el complemento definido por el notebook. La inferencia agregada procesa, por tanto, 15 modelos/folds.

### Inferencia y métricas

La inferencia carga los mejores pesos de cada fold y procesa cada señal completa en ventanas de 20 segundos con stride de 10 segundos, a una frecuencia asumida de 200 Hz. Conserva las zonas centrales de las ventanas para evitar duplicar predicciones.

Después aplica:

- `sigmoid` sobre los logits.
- Umbral `0.5` para la detección y `0.425` para la duración del evento.
- Unión de detecciones cercanas.
- Filtro de duración entre 0.3 y 5 segundos.
- Emparejamiento de eventos mediante IoU.
- Microaverage sobre los folds, usando IoU `0.2` para seleccionar el punto reportado.

El resultado registrado en [`inference_results_moda_microaverage.json`](eeg_code/inference_results_moda_microaverage.json) contiene 18.577 resultados, F1 de 82.5653%, precisión de 83.6439% y recall de 81.5141%. Es un artefacto de ejecución del notebook, no una validación clínica ni una garantía de reproducibilidad sin los mismos datos, pesos y entorno.

### Tutorial de incertidumbre

[`uncertainty_tutorial.ipynb`](uncertainty_tutorial.ipynb) es un experimento independiente sobre datos sintéticos. Usa un MLP flexible con conexiones residuales cuando las dimensiones coinciden y compara:

- **Ensemble:** entrena cinco MLP y calcula media, varianza y entropía de las probabilidades.
- **MC Dropout:** hace 20 pasadas estocásticas con dropout activo.
- **Modelo individual:** usa `1 - max(probabilidad)` como score de incertidumbre.

El ejemplo genera un dataset de espiral de tres brazos, agrega un clúster anómalo, estandariza los datos y usa una división entrenamiento/prueba de 70/30. También incluye una alternativa comentada con dos lunas. Las visualizaciones muestran regiones de predicción e incertidumbre, pero todavía no constituyen una implementación EEG de Laplace, Conformal Prediction, Neural Processes o ensembles temporales.

Nota: la función `max_prob_metric` devuelve valores mayores cuando la confianza es menor, aunque su comentario interno dice lo contrario. Mantener esta convención explícita al comparar métricas.

## Lineamientos del repositorio

- Solicitar al tutor la base de datos, los pesos y cualquier artefacto protegido antes de ejecutar el pipeline completo.
- No incorporar al repositorio datos médicos, identificadores de pacientes, señales HDF5, pesos privados ni salidas que permitan reconstruir información sensible.
- Mantener las particiones a nivel de sujeto. No mezclar ventanas del mismo sujeto entre entrenamiento, validación y prueba.
- Registrar siempre la evaluación, el fold, el dataset utilizado, los umbrales y si las métricas son por fold o microagregadas.
- Separar los experimentos sintéticos del tutorial de los resultados EEG; el tutorial sirve para explorar conceptos de incertidumbre, no para casos clínicos.
- Comparación de métodos debe incluir idealmente calibración, detección de errores, cobertura/rechazo y costo computacional.
- Documentar cualquier cambio en resolución, frecuencia de muestreo, ventanas, stride, thresholds, normalización, augmentación o protocolo de folds.

## Referencias

[1] E. Daxberger, A. Kristiadi, A. Immer, R. Eschenhagen, M. Bauer, and P. Hennig, “Laplace redux—Effortless Bayesian deep learning,” *Advances in Neural Information Processing Systems*, vol. 34, pp. 20089–20103, 2021.

[2] A. N. Angelopoulos and S. Bates, “Conformal prediction: A gentle introduction,” *Foundations and Trends in Machine Learning*, vol. 16, no. 4, pp. 494–591, 2023.

[3] M. Garnelo et al., “Conditional neural processes,” in *Proc. 35th International Conference on Machine Learning (ICML)*, vol. 80, pp. 1704–1713, 2018.

[5] J. Harrison, J. Willes, y J. Snoek, “Variational Bayesian last layers,” ICLR, 2024.

[6] H. Kim, A. Mnih, J. Schwarz, M. Garnelo, A. Eslami, D. Rosenbaum, O. Vinyals, and Y. W. Teh, “Attentive neural processes,” in *Proc. International Conference on Learning Representations (ICLR)*, 2019.

[7] B. Lakshminarayanan, A. Pritzel, y C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” NeurIPS, 2017.

[8] N. I. Tapia-Rivas, P. A. Estévez, and J. A. Cortes-Briones, “A robust deep learning detector for sleep spindles and K-complexes: Towards population norms,” *Scientific Reports*, vol. 14, Art. no. 263, 2024.
