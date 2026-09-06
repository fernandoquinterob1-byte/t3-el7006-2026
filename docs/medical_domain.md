# SEED, dominio médico y protocolo de detección

La referencia clínica y metodológica de este proyecto es Tapia-Rivas, Estévez y Cortes-Briones (2024), *A robust deep learning detector for sleep spindles and K-complexes: towards population norms*, [Scientific Reports 14, 263](https://doi.org/10.1038/s41598-023-50736-7).

## Problema que resuelve SEED

SEED (*Sleep EEG Event Detector*) detecta eventos transitorios en EEG de sueño: huso del sueño (*sleep spindle*, SS) o complejo K (KC). El artículo sitúa ambos patrones principalmente en sueño no-REM N2; estudia su utilidad para análisis de población y como posibles marcadores, no como diagnóstico autónomo. Los husos y los complejos K son patrones EEG breves relacionados con cognición, aprendizaje, memoria y procesamiento de estímulos.

Para el artefacto MODA de este repositorio, el código legacy solamente contiene el postprocesamiento `process_spindle_simple`; por ello el objetivo práctico confirmado aquí es **detección de husos**. Las etiquetas HDF5 son una máscara binaria temporal: 1 significa que el instante pertenece al evento anotado y 0 que pertenece al fondo. No existe una etiqueta multiclase SS/KC en los archivos ni en el notebook.

## Entrada, salida y eventos

El artículo usa EEG de un solo canal C3 o C4 durante N2, normalizado a 200 Hz, y ventanas de 20 s. En el HDF5 privado auditado, las ventanas aumentadas tienen 4000 muestras, consistentes con 20 s a la frecuencia asumida de 200 Hz; las señales sin aumento tienen 23 000 muestras (115 s). La red produce una salida binaria temporal: una probabilidad por timestep de pertenecer al evento. El notebook PyTorch expande sus logits a resolución de entrada por repetición de factor 8 y luego aplica `sigmoid`.

Para inferencia legacy, las señales completas se procesan con ventanas de 20 s separadas 10 s y se conservan las regiones centrales. Las probabilidades se umbralizan con 0.5 para existencia y 0.425 para duración, se unen detecciones cercanas y se elimina/recorta según la lógica exacta de `process_spindle_simple`. Esa lógica aplica duración mínima de 0.3 s y máxima de 5 s. El artículo describe para SS una regla relacionada, pero con máximo de 3 s; para reproducir el F1 histórico de este repositorio prevalece el notebook.

## Matching y métricas de eventos

El notebook extrae intervalos reales y predichos, los empareja usando su algoritmo secuencial y representa un falso positivo como -1 y un falso negativo como 0. Para un par de intervalos, calcula `IoU(A, B) = |A ∩ B| / |A ∪ B|`.

En el punto de reporte histórico, un par con IoU >= 0.2 es TP; una anotación no detectada es FN y una detección sin anotación correspondiente es FP. Luego agrega todos los eventos de los folds (microaverage) para precisión, recall y F1. Un TP localiza y encuentra un evento anotado; un FP agrega un evento que podría requerir revisión clínica; un FN omite un evento relevante. IoU mayor indica inicio/final temporal más precisos.

## Información por confirmar

- Si el conjunto privado entregado para el curso contiene únicamente SS o también una variante KC; el nombre del HDF5 y el notebook sólo permiten confirmar el flujo SS.
- La trazabilidad exacta entre el filtrado/normalización/augmentación descritos en el artículo y los HDF5 ya preparados. El notebook no vuelve a aplicar esos pasos.
- La razón del máximo de 5 s del notebook frente a los 3 s para SS indicados en el artículo; no debe cambiarse sin indicación del tutor.
- Si debe aplicarse una máscara N2 adicional: el HDF5 no expone un dataset de máscara y el notebook no la usa explícitamente.
