# Resumen corto de métodos de incertidumbre

Estos métodos son candidatos futuros; ninguno está implementado aún. El baseline mantiene SEED sin modificar y usa sólo entropía/confianza post-hoc.

## Laplace Redux / last-layer Laplace

La aproximación de Laplace aproxima el posterior alrededor del MAP mediante una gaussiana cuya curvatura proviene de la loss. En una versión *last-layer*, el backbone SEED queda fijo y sólo la capa final obtiene incertidumbre paramétrica: es mucho más barata y apunta principalmente a incertidumbre epistémica. Puede subestimar la incertidumbre si el resto de la red cambia mucho entre regiones del EEG. Referencia: [Daxberger et al. (2021)](https://arxiv.org/abs/2106.14806).

## Conformal prediction

Usa un conjunto de calibración y scores de no-conformidad para construir conjuntos o decisiones con garantía de cobertura bajo intercambiabilidad. En SEED podría calibrar timesteps o eventos y apoyar rechazo o revisión, sin necesitar una red bayesiana. Sus garantías pueden degradarse por cambio de distribución o dependencia temporal. Referencia: [Angelopoulos y Bates (2023)](https://arxiv.org/abs/2107.07511).

## Conditional y Attentive Neural Processes

Un Conditional Neural Process aprende de pares contexto→objetivo una distribución predictiva, emulando ideas de procesos estocásticos. Attentive NP agrega atención para evitar que una representación global pierda detalles del contexto. Podrían modelar variabilidad entre señales o sujetos EEG, pero son más complejos de entrenar y no son un reemplazo inmediato reproducible de SEED. Referencias: [CNP](https://arxiv.org/abs/1807.01622) y [ANP](https://arxiv.org/abs/1901.05761).

## Variational Bayesian Last Layers

Mantiene un extractor determinista y aproxima variacionalmente una distribución en la última capa. Es una alternativa escalable a hacer toda SEED bayesiana y captura incertidumbre de pesos de la cabeza; depende de que las features fijas sean adecuadas fuera de distribución. Referencia: [Harrison et al.](https://openreview.net/forum?id=Sg3jrBxdXo).

## Deep Ensembles

Entrena varias instancias independientes y promedia sus probabilidades; la dispersión entre modelos aporta un proxy de incertidumbre epistémica. Suele ser robusto y simple conceptualmente, pero cuesta varios entrenamientos completos de SEED. Referencia: [Lakshminarayanan et al. (2017)](https://arxiv.org/abs/1612.01474).

## Temporal shifted windows

Aplica el mismo detector a ventanas temporalmente desplazadas de una señal y compara o combina las salidas alineadas. Puede revelar sensibilidad a bordes y a pequeños shifts, una propiedad directamente relevante para eventos EEG breves. No sustituye una distribución posterior y debe preservar la regla legacy de stitching y postprocesamiento al evaluar eventos.
