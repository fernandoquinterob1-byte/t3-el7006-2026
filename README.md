# Estimación de incertidumbre en redes neuronales aplicada a medicina (EEG)

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
