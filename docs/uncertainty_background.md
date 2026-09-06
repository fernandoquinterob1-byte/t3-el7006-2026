# Incertidumbre para la segmentación temporal binaria de SEED

## Aleatoria y epistémica

La incertidumbre **aleatoria** representa ruido o ambigüedad inherente a la medición o anotación. En EEG puede provenir de ruido, transiciones poco claras de un huso o desacuerdo entre expertos; no desaparece necesariamente al reunir más datos. La incertidumbre **epistémica** es falta de conocimiento del modelo: regiones poco observadas, sujetos o artefactos fuera de distribución. Puede reducirse con datos representativos adicionales.

La predicción bayesiana se escribe como `p(y | x, D) = ∫ p(y | x, θ) p(θ | D) dθ`. La epistémica aparece en la distribución posterior sobre parámetros `p(θ | D)`. Un detector determinista sólo entrega una aproximación puntual; ensembles o aproximaciones posteriores permiten estimarla después.

## Métricas del MVP binario temporal

Para cada timestep, con `p_t = sigmoid(logit_t)`: `confidence_t = max(p_t, 1 - p_t)`, `uncertainty_t = 1 - confidence_t` y `H_t = -p_t log(p_t + ε) - (1-p_t) log(1-p_t + ε)`.

Entropía baja significa una probabilidad concentrada cerca de 0 o 1; entropía alta, decisión ambigua cerca de 0.5. El módulo actual calcula estas cantidades por timestep y resume media/mediana para predicciones correctas e incorrectas. Para detectar errores, el target es `1[y_pred != y_true]` y el score es `uncertainty`; AUROC sólo se informa si ambas clases de error están presentes.

Para modelos con muestras de parámetros, la información mutua es `MI[y, θ | x, D] = H[E_{p(θ|D)} p(y|x,θ)] - E_{p(θ|D)} H[p(y|x,θ)]`. Es un indicador de componente epistémica y no se calcula todavía, porque este MVP usa una única pasada determinista.

## Análisis selectivo futuro

Risk-coverage ordena predicciones por incertidumbre: *coverage* es la fracción aceptada y *selective risk* es el error dentro de las aceptadas. El diseño guarda salidas temporales y límites de eventos para poder estudiar después incertidumbre temporal vs. error, incertidumbre de evento vs. IoU, y TP vs. FP. La primera agregación de evento disponible será **media de entropía dentro del intervalo predicho**, explícitamente exploratoria.
