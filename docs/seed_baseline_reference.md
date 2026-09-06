# Contraste de referencia: artículo, notebook y módulo nuevo

| Aspecto | Artículo Tapia-Rivas et al. (2024) | `eeg_code/run_model.ipynb` | `src/eeg_uq/` | Decisión |
| --- | --- | --- | --- | --- |
| Unidad de salida | Segmentación binaria temporal, descrita como dos probabilidades/softmax a resolución /8 | Un logit binario por timestep reducido, repetido x8; `BCEWithLogitsLoss` | Misma arquitectura/logit/BCE del notebook | Preservar notebook para el baseline del curso. |
| Contexto de entrada | 20 s puntuables + 520 muestras por borde | HDF5 de 4000 muestras; no concatena bordes en el notebook | No añade bordes | No inventar bordes: usar HDF5 y notebook. Confirmar preprocesamiento. |
| Preprocesamiento | MODA 0.3–30 Hz, remuestreo 200 Hz, normalización y clipping; augmentación descrita | Consume HDF5 preprocesado; `norm_data=False`; no expone filtrado/máscara | Consume exactamente esos HDF5 | El HDF5 es la frontera práctica. |
| Postprocesamiento SS | Ajuste de probabilidad, umbrales, unión; mínimo 0.3 s, máximo 3 s | Umbrales 0.5/0.425; unión; mínimo 0.3 s, máximo 5 s | Funciones extraídas literalmente | Preservar 5 s para reproducibilidad. |
| Métricas | IoU, TP si IoU >= 0.2, microaverage en MODA | Pairing secuencial propio; FP=-1/FN=0; curva de 100 IoU y punto más cercano a 0.2 | Misma representación y curva | Preservar exactamente. |
| Folds | 3 repeticiones x 5 folds, sin solapamiento de sujetos | `eval_1..3` / `fold_1..5`; train=complemento val/test | Misma estructura; la CLI usa índices 0–2/0–4 y mapea internamente | Nunca aplanar silenciosamente. |

El código modular no pretende ser una nueva implementación del paper: es una interfaz reproducible para el artefacto legacy. El único agregado experimental es exportar salidas y analizarlas post-hoc; no modifica la red ni sus eventos.

## Resultado histórico que se debe verificar después

El JSON versionado registra F1 = 82.5653%, precisión = 83.6439% y recall = 81.5141% con 18 577 resultados, usando los 15 pares evaluation/fold. Es una referencia de reproducción, no un objetivo para ajustar umbrales o código.
