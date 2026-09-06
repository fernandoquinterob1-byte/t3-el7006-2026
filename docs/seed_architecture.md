# Arquitectura SEED usada en este repositorio

La definición ejecutable de referencia es `SEED_Model` en
`eeg_code/run_model.ipynb`; `src/eeg_uq/model.py` es una transliteración de esa
definición con nombres Python convencionales. Ambas usan:

1. Entrada `(batch, tiempo)` seguida de `BatchNorm1d(1)`.
2. Un `ConvBlock` con dos `Conv1d` kernel 3, BatchNorm y ReLU, seguido de
   average pooling x2. La configuración concreta es `embed_dim=64`, `blocks=1`.
3. Dos `StageMultiDilatedConvolutions`: cada Conv MDB concatena cuatro ramas,
   cada rama con dos convoluciones kernel 3, BatchNorm y ReLU, con dilataciones
   1, 2, 4 y 8. Cada bloque es seguido de average pooling x2.
4. Dos capas `BiLSTM` de 256 unidades por dirección, con dropout 0.2 antes y
   0.5 entre capas.
5. `RED_FC`: dropout 0.5, capa lineal 512→128, ReLU y capa lineal 128→1. El
   sesgo inicial de la salida corresponde a probabilidad positiva 0.1.
6. Repetición temporal x8 de los logits. La salida final es `(batch, tiempo)`
   de logits binarios; `sigmoid` se aplica fuera del modelo.

El artículo describe la misma jerarquía conceptual: *local encoding* mediante
convoluciones/Conv MDB, *contextualization* mediante BLSTM y clasificación
densa. Habla de una capa Conv1D final y softmax de dos clases a resolución /8;
el notebook usa una `Linear` equivalente a una operación 1×1 sobre features y
un único logit entrenado con `BCEWithLogitsLoss`, luego lo repite x8. No se
intentó reconciliar esas formulaciones: el módulo nuevo preserva la del
notebook, que además es la compatible con los checkpoints entregados.

No hay MC-Dropout activo ni cabezas bayesianas en esta arquitectura. Los
dropouts existentes son regularización de entrenamiento y la inferencia legacy
usa `model.eval()`.
