# Conformal Prediction post-hoc sobre SEED

## Qué hace

No modifica ni reentrena SEED. El flujo es:

```text
EEG → SEED ya entrenado → probabilidades temporales
    → calibration (validation) → q_hat → prediction sets en test
```

Para cada instante, SEED entrega `p = P(y=1|x)`. Entonces:

```text
P(y=1|x) = p
P(y=0|x) = 1-p
s(x,y)   = 1-P(y|x)
```

En particular, si la etiqueta verdadera es evento (`y=1`), el score es `1-p`;
si es fondo (`y=0`), el score es `p`.

## Calibration y garantía

El conjunto **validation** del fold se reutiliza inicialmente como calibration
set, tal como se solicitó para el esquema 3 train / 1 validation / 1 test.
Sólo se incluyen timesteps de la máscara `valid_mask` o `annotation_mask`; no
padding, contexto artificial ni regiones ignoradas por el protocolo.

Para `n` scores válidos y nivel de error `alpha`, se usa el rango de muestra
finita estándar:

```text
k     = ceil((n + 1) (1 - alpha))
q_hat = k-ésimo score ordenado, con k limitado a n
```

No equivale a calcular simplemente el percentil 95 con `np.quantile`. Bajo los
supuestos usuales de intercambiabilidad, la cobertura marginal temporal es:

```text
P(Y pertenece a Gamma(X)) >= 1 - alpha
```

La implementación registra una advertencia metodológica: validation también
selecciona el mejor checkpoint mediante early stopping, por lo que no es una
calibración totalmente independiente. El test no entra nunca en `q_hat`.

## Prediction set

En cada timestep del test:

```text
Gamma(x) = {c en {0,1}: 1 - P(c|x) <= q_hat}
```

El NPZ guarda arrays numéricos, no objetos:

| Set | Interpretación | Codificación |
| --- | --- | --- |
| `{0}` | fondo conforme | `set_size=1`, sólo `contains_background` |
| `{1}` | evento conforme | `set_size=1`, sólo `contains_event` |
| `{0,1}` | ambiguo | `set_size=2` |
| `{}` | ninguna clase conforme | `set_size=0` |

Conformal entrega conjuntos calibrados y una garantía de cobertura bajo sus
supuestos; no debe interpretarse como incertidumbre epistémica.

## Métricas

Sobre timesteps válidos del test se guardan:

- cobertura empírica y cobertura objetivo `1-alpha`;
- tamaño medio del conjunto;
- tasa singleton, ambigua y vacía.

También se preparan resúmenes exploratorios por evento: probabilidad SEED (p75
dentro del evento), fracción ambigua, tamaño medio del set, fracción singleton,
IoU y estado TP/FP. La curva de rechazo prueba porcentajes de ambigüedad 0.10,
0.25 y 0.50, y muestra aceptación frente a F1, precisión e IoU.

No hay todavía una garantía conformal para eventos: la garantía marginal de
timesteps no se transfiere automáticamente a IoU, TP/FP ni eventos completos.

## Diferencia con otros métodos

Conformal usa una calibration set y probabilidades ya producidas por SEED para
formar conjuntos con cobertura. Laplace aproxima incertidumbre sobre
parámetros; un Temporal Ensemble mide estabilidad entre predicciones de
ventanas. Son señales distintas y módulos separados.

Los comandos y la ubicación de los resultados están centralizados en
[`workflow.md`](workflow.md).
