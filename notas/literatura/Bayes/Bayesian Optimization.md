# Bayesian Optimization

## Definición

Bayesian Optimization (BO) es una **técnica de optimización basada en modelos probabilísticos** diseñada para encontrar el óptimo de funciones objetivo que son:

- **Costosas de evaluar** (ej: entrenar un modelo de deep learning por horas)
- **Sin gradientes disponibles** (black-box functions)
- **Con ruido en las observaciones**
- **De alta dimensionalidad limitada** (típicamente < 20 dimensiones)

En lugar de probar al azar (Random Search) o exhaustivamente (Grid Search) todas las configuraciones de hiperparámetros, BO construye un **modelo estadístico aproximado** (surrogate model) de la función objetivo y usa este modelo para decidir inteligentemente qué configuraciones evaluar a continuación.

**Ventaja clave**: Encuentra hiperparámetros óptimos con **menos evaluaciones** → más sample-efficient.

---

## Motivación: Problema de Optimización de Hiperparámetros

En Machine Learning, antes de entrenar un modelo, se deben fijar **hiperparámetros**:

- Learning rate ($\alpha$)
- Número de capas
- Batch size
- Número de árboles (en Random Forest)
- Regularización ($\lambda$)
- etc.

**Objetivo**: Encontrar la configuración que maximice una métrica (accuracy, F1-score, etc.) o minimice un error (loss, RMSE, etc.).

---

## Métodos Tradicionales vs Bayesian Optimization

### 1. Grid Search

**Funcionamiento**:
- Recorre TODAS las combinaciones posibles en una "rejilla" predefinida
- Garantiza probar todo el espacio discretizado

**Ventajas**:
- Simple de implementar
- Reproducible
- No requiere teoría sofisticada

**Desventajas**:
- ❌ Explota en complejidad: $O(d^n)$ donde $d$ = valores por dimensión, $n$ = número de hiperparámetros
- ❌ Muy lento: desperdicia tiempo en configuraciones malas
- ❌ No aprende de evaluaciones previas

**Ejemplo**:
```python
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64, 128]
# Total: 3 × 4 = 12 evaluaciones
```

### 2. Random Search

**Funcionamiento**:
- Escoge combinaciones **al azar** del espacio de búsqueda
- Evalúa un presupuesto fijo de configuraciones

**Ventajas**:
- Suele encontrar algo decente más rápido que Grid Search
- Mejor en espacios de alta dimensión
- Fácil de paralelizar

**Desventajas**:
- ❌ Aún no usa información previa → sigue siendo "ciego"
- ❌ No balancea exploración vs explotación
- ❌ Puede probar configuraciones muy similares por azar

### 3. Bayesian Optimization ✅

**Funcionamiento**:
- **Aprende de lo ya probado**
- Usa un modelo probabilístico que balancea:
  - **Explotación**: Probar cerca de los mejores hiperparámetros encontrados
  - **Exploración**: Probar en zonas poco exploradas que podrían ser mejores

**Ventajas**:
- ✅ Sample-efficient: Menos evaluaciones para encontrar el óptimo
- ✅ Usa información acumulada inteligentemente
- ✅ Cuantifica incertidumbre en predicciones
- ✅ Adaptable a restricciones y costos

**Desventajas**:
- Más complejo de implementar
- Overhead computacional del surrogate model
- No siempre mejor que Random Search en alta dimensión (> 20D)

---

## ¿Cómo Funciona Bayesian Optimization?

Bayesian Optimization tiene **3 componentes fundamentales**:

1. **Surrogate Model** (Modelo Sustituto)
2. **Acquisition Function** (Función de Adquisición)
3. **Optimization Loop** (Bucle Iterativo)

![Figura: Diagrama de flujo de Bayesian Optimization]
<!-- TODO: Agregar diagrama de flujo BO -->

---

## 1. Surrogate Model

### Definición

Es un **modelo probabilístico barato** que intenta predecir el valor de la función objetivo real $f(x)$ sin tener que evaluarla.

**Características clave**:
- No solo predice un valor → también dice **cuánta confianza** tiene (incertidumbre)
- Se actualiza con cada nueva observación
- Es computacionalmente eficiente comparado con entrenar el modelo real

### Tipos de Surrogate Models

#### A) Gaussian Process (GP) 🔵

**¿Qué es?**

Un GP es un modelo probabilístico que genera una **distribución sobre funciones suaves**. Dado un conjunto de puntos evaluados, un GP predice:
- La **media** $\mu(x)$: mejor estimación de $f(x)$
- La **varianza** $\sigma^2(x)$: incertidumbre en la predicción

**Formulación matemática**:

$$f(x) \sim \text{GP}(\mu(x), \kappa(x, x'))$$

Donde:
- $\mu(x)$: función de media (usualmente se asume 0)
- $\kappa(x, x')$: **función kernel** que mide la covarianza entre puntos

**Kernels comunes**:

1. **RBF (Radial Basis Function)**:
   $$\kappa(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$
   - Asume que puntos cercanos tienen valores similares
   - $\ell$: length-scale (controla qué tan rápido decae la correlación)

2. **Matérn**:
   $$\kappa(x, x') = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\|x-x'\|}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\|x-x'\|}{\ell}\right)$$
   - Más flexible que RBF
   - $\nu$ controla la suavidad

**Intuición visual**:

> Imagina que pintas una **curva suave** sobre los puntos que ya probaste. A medida que te alejas de puntos conocidos, la "pintura se vuelve borrosa" → **mayor incertidumbre**.

![Figura: GP con media y banda de incertidumbre]
<!-- TODO: Agregar figura de GP mostrando μ(x) ± 2σ(x) -->

**Ventajas**:
- ✅ Excelente estimación de incertidumbre
- ✅ Ideal para acquisition functions (exploración vs explotación)
- ✅ Funciona muy bien en espacios de baja dimensión (< 20 hiperparámetros)
- ✅ Si la función es suave y continua, muy eficiente
- ✅ Solidez matemática y teórica

**Desventajas**:
- ❌ No escala bien: $O(n^3)$ con número de puntos evaluados
- ❌ Problemas en alta dimensionalidad (> 20D)
- ❌ Asume suavidad: puede fallar en funciones discontinuas
- ❌ Selección de kernel puede afectar rendimiento

#### B) Tree-structured Parzen Estimator (TPE) 🌳

**¿Qué es?**

En vez de modelar directamente $f(x)$, TPE modela la **distribución de configuraciones** que dieron:
- Resultados **buenos** → $p(x | y < y^*)$
- Resultados **malos** → $p(x | y \geq y^*)$

Luego busca nuevos puntos en regiones con alta densidad de configuraciones "buenas".

**Formulación**:

$$p(x | y) =
\begin{cases}
\ell(x) & \text{si } y < y^* \text{ (buenos)}\\
g(x) & \text{si } y \geq y^* \text{ (malos)}
\end{cases}$$

Donde $y^*$ es un percentil (ej: top 15% de resultados).

**Ventajas**:
- ✅ Escala mucho mejor a espacios de mayor dimensión
- ✅ Funciona bien con hiperparámetros **categóricos** y **mixtos** (enteros, booleanos, listas)
- ✅ Implementado en librerías populares (Hyperopt)
- ✅ Menos costoso computacionalmente que GP

**Desventajas**:
- ❌ Estimación de incertidumbre menos precisa que GP
- ❌ Menos "teórico" → más heurístico
- ❌ No aprovecha estructura suave de la función

#### C) Adaptive Tree-structured Parzen Estimator (ATPE) 🔄

**¿Qué es?**

Una versión **mejorada de TPE** que ajusta automáticamente sus hiperparámetros internos:
- Percentiles que definen "bueno" vs "malo"
- Parámetros de las distribuciones $\ell(x)$ y $g(x)$

**Ventajas**:
- ✅ Más robusto que TPE en distintos escenarios
- ✅ Encuentra mejores soluciones más rápido
- ✅ Auto-tuning de parámetros internos

**Desventajas**:
- ❌ Más complejo de entender
- ❌ Todavía heurístico (no tiene solidez matemática de GP)

#### D) Otros Surrogate Models

- **Random Forest**: Ensemble de árboles de decisión
- **Gradient Boosted Trees**: XGBoost, LightGBM
- **Neural Networks**: Para espacios muy complejos

---

## 2. Acquisition Function

### Definición

Dada la predicción del surrogate model, la **acquisition function** $\alpha(x)$ define una **regla** para elegir el siguiente punto $x_{next}$ a evaluar con la función objetivo real.

**Objetivo**: Balancear **explotación** vs **exploración**

$$x_{next} = \arg\max_{x} \alpha(x)$$

### Principales Acquisition Functions

#### A) Probability of Improvement (PI)

**Idea**: Maximizar la probabilidad de mejorar el mejor valor observado hasta ahora.

$$\text{PI}(x) = P(f(x) > f(x^+)) = \Phi\left(\frac{\mu(x) - f(x^+)}{\sigma(x)}\right)$$

Donde:
- $f(x^+)$ = mejor valor observado
- $\Phi$ = función CDF de la normal estándar

**Característica**: Muy conservadora (tiende a explotar)

#### B) Expected Improvement (EI) ⭐ [MÁS USADA]

**Idea**: Maximizar la **mejora esperada** sobre el mejor valor.

$$\text{EI}(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)] =
\begin{cases}
(\mu(x) - f(x^+))\Phi(Z) + \sigma(x)\phi(Z) & \text{si } \sigma(x) > 0\\
0 & \text{si } \sigma(x) = 0
\end{cases}$$

Donde $Z = \frac{\mu(x) - f(x^+)}{\sigma(x)}$

**Ventaja**: Buen balance entre exploración y explotación

#### C) Upper Confidence Bound (UCB)

**Idea**: Optimismo ante la incertidumbre.

$$\text{UCB}(x) = \mu(x) + \kappa \cdot \sigma(x)$$

Donde $\kappa$ controla el balance exploración/explotación:
- $\kappa$ alto → más exploración
- $\kappa$ bajo → más explotación

**Característica**: Simple y efectiva

#### D) Thompson Sampling

**Idea**: Muestrear funciones del posterior del GP y optimizar la muestra.

**Ventaja**: Naturalmente estocástico → bueno para paralelización

---

## 3. Bucle Iterativo (Sequential Model-Based Optimization - SMBO)

### Algoritmo General

```
1. Definir espacio de búsqueda S
2. Inicializar con n_init evaluaciones aleatorias
3. FOR iteración t = n_init+1 to n_max:
   a) Entrenar surrogate model con datos observados
   b) Optimizar acquisition function para encontrar x_next
   c) Evaluar f(x_next) con función objetivo real
   d) Añadir (x_next, f(x_next)) al conjunto de datos
4. RETURN mejor configuración encontrada
```

### Flujo Detallado

#### Paso 1: Definir Search Space

El usuario define:
- Rangos para hiperparámetros continuos: $x_i \in [a, b]$
- Opciones para categóricos: $x_j \in \{\text{Adam}, \text{SGD}, \text{RMSprop}\}$
- Distribuciones: uniforme, log-uniforme, normal, etc.

**Ejemplo**:
```python
space = {
    'learning_rate': hp.loguniform('lr', np.log(1e-5), np.log(1e-1)),
    'batch_size': hp.choice('bs', [16, 32, 64, 128]),
    'n_layers': hp.quniform('layers', 2, 10, 1),
    'activation': hp.choice('act', ['relu', 'tanh', 'elu'])
}
```

#### Paso 2: Random Sampling Inicial

Se toman $n_{init}$ (típicamente 5-10) configuraciones **aleatorias** para:
- Construir un historial inicial $\mathcal{H} = \{(x_1, y_1), \ldots, (x_{n_{init}}, y_{n_{init}})\}$
- Dar información diversa al surrogate model

**¿Por qué aleatorio?**
- Sin evaluaciones previas, no hay información para guiar la búsqueda
- Evita sesgo inicial

#### Paso 3: Construcción del Modelo Probabilístico

Con el historial $\mathcal{H}$, se entrena el surrogate model:

$$p(y \mid x, \mathcal{H}) \approx p_{\text{surrogate}}(y \mid x)$$

**Para GP**:
- Se ajustan hiperparámetros del kernel (length-scale, variance)
- Se calcula la distribución posterior: $p(f \mid \mathcal{H})$

**Para TPE**:
- Se dividen observaciones en "buenas" ($y < y^*$) y "malas" ($y \geq y^*$)
- Se estiman $\ell(x)$ y $g(x)$ usando kernel density estimation

#### Paso 4: Optimización de Acquisition Function

Se resuelve:

$$x_{next} = \arg\max_{x \in S} \alpha(x \mid \mathcal{H})$$

**Métodos de optimización**:
- Para espacios continuos: L-BFGS, DIRECT, CMA-ES
- Para espacios mixtos: Grid search sobre acquisition, evolutionary algorithms

#### Paso 5: Evaluación Real

- Se entrena el modelo con configuración $x_{next}$
- Se mide la métrica objetivo: $y_{next} = f(x_{next})$
- **Esta es la parte costosa** (puede tomar horas)

#### Paso 6: Actualización del Historial

$$\mathcal{H} \leftarrow \mathcal{H} \cup \{(x_{next}, y_{next})\}$$

El surrogate model se vuelve **más preciso** con cada iteración.

![Figura: Evolución del GP a través de iteraciones]
<!-- TODO: Agregar animación o secuencia de GPs actualizándose -->

---

## Preguntas Frecuentes

### ¿Cuántas muestras iniciales necesito?

**Recomendación general**:
- Espacios simples (< 5 dim): $n_{init} = 5$
- Espacios medianos (5-10 dim): $n_{init} = 10-20$
- Espacios complejos (> 10 dim): $n_{init} = 50-100$

**Regla empírica**: $n_{init} \approx 2 \times d$ donde $d$ = dimensionalidad

### ¿Cómo establecer los límites del search space?

**Estrategias**:

1. **Experiencia previa**: Usar valores típicos de la literatura
2. **Órdenes de magnitud**: Para learning rate: $[10^{-5}, 10^{-1}]$
3. **Escalas logarítmicas**: Para parámetros que varían exponencialmente
4. **Restricciones físicas**: Batch size debe ser potencia de 2 para eficiencia

**Cuidado**:
- Si el límite es muy estrecho → puede excluir el óptimo
- Si es muy amplio → necesitas más evaluaciones

### ¿Cuándo detener la optimización?

**Criterios de parada**:

1. **Presupuesto fijo**: $n_{max}$ evaluaciones
2. **Convergencia**: No hay mejora en $k$ iteraciones consecutivas
3. **Tiempo límite**: Wall-clock time
4. **Objetivo alcanzado**: $f(x) >$ umbral deseado

---

## Comparación de Surrogate Models

| Característica | Gaussian Process | TPE | ATPE | Random Forest |
|---------------|------------------|-----|------|---------------|
| **Incertidumbre** | Excelente | Buena | Buena | Moderada |
| **Dimensionalidad** | Baja (< 20) | Media-Alta | Media-Alta | Media |
| **Categóricos** | Difícil | Excelente | Excelente | Bueno |
| **Complejidad** | $O(n^3)$ | $O(n \log n)$ | $O(n \log n)$ | $O(nt \log n)$ |
| **Interpretabilidad** | Alta (teórico) | Media (heurístico) | Baja | Media |
| **Sample Efficiency** | Muy Alta | Alta | Muy Alta | Media |

**Recomendación por caso**:

- **Funciones suaves, < 10 dim, budget pequeño**: Gaussian Process con EI
- **Espacios mixtos, > 10 dim, categóricos**: TPE o ATPE
- **Alta dimensionalidad (> 20 dim), muchos evaluaciones**: Random Forest + UCB
- **Paralelización masiva**: Thompson Sampling

---

## Herramientas en Python

### 1. Optuna ⭐ [RECOMENDADO]

**URL**: https://optuna.org/

**Características**:
- Interfaz moderna y limpia
- TPE como default (escalable)
- Pruning automático de trials malos
- Visualizaciones interactivas
- Paralelización nativa
- Integración con frameworks (PyTorch, TensorFlow, Keras)

**Ejemplo**:
```python
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
```

### 2. Hyperopt

**URL**: https://hyperopt.github.io/hyperopt/

**Características**:
- Implementación original de TPE
- Muy usado en Kaggle
- Espacios de búsqueda flexibles
- MongoDB para paralelización

**Ejemplo**:
```python
from hyperopt import hp, fmin, tpe, Trials

space = {
    'x': hp.uniform('x', -10, 10),
    'y': hp.choice('y', [1, 2, 3])
}

best = fmin(
    fn=lambda params: (params['x'] - 2)**2,
    space=space,
    algo=tpe.suggest,
    max_evals=100
)
```

### 3. scikit-optimize

**URL**: https://scikit-optimize.github.io/stable/

**Características**:
- Gaussian Process como default
- API estilo scikit-learn
- Varias acquisition functions (EI, PI, LCB)
- Diagnóstico y visualización

**Ejemplo**:
```python
from skopt import gp_minimize

def objective(params):
    x, y = params
    return x**2 + y**2

result = gp_minimize(
    objective,
    [(-5.0, 5.0), (-5.0, 5.0)],
    n_calls=50,
    random_state=42
)
```

### 4. Ray Tune

**URL**: https://docs.ray.io/en/latest/tune/index.html

**Características**:
- Escalabilidad a clusters
- Early stopping automático
- Integración con Ray (paralelización distribuida)
- Soporte para múltiples algoritmos (Optuna, HyperOpt, Ax)

---

## Ejemplo Completo: Optimizar Hiperparámetros de Red Neuronal

```python
import optuna
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Cargar datos
X, y = load_digits(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

def objective(trial):
    # Hiperparámetros a optimizar
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_units = trial.suggest_int('n_units', 32, 256)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

    # Construir modelo
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(64,)))

    for i in range(n_layers):
        model.add(keras.layers.Dense(n_units, activation='relu'))
        model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(10, activation='softmax'))

    # Compilar
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entrenar
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=0
    )

    # Retornar métrica a MAXIMIZAR (Optuna minimiza por default)
    return history.history['val_accuracy'][-1]

# Crear estudio (maximización)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Resultados
print(f"Mejor accuracy: {study.best_value:.4f}")
print(f"Mejores hiperparámetros: {study.best_params}")
```

---

## Ventajas y Limitaciones

### Ventajas ✅

1. **Sample efficiency**: Encuentra buenos hiperparámetros con ~20-100 evaluaciones (vs miles en Grid/Random)
2. **Adaptativo**: Aprende de evaluaciones previas
3. **Cuantifica incertidumbre**: Sabe dónde tiene confianza y dónde no
4. **Flexible**: Espacios mixtos (continuos, discretos, categóricos)
5. **Teóricamente fundamentado**: Garantías de convergencia (bajo ciertas condiciones)

### Limitaciones ❌

1. **Alta dimensionalidad**: Sufre en > 20-50 dimensiones (curse of dimensionality)
2. **Overhead computacional**: El surrogate puede ser costoso de entrenar
3. **Asunciones de suavidad**: GP asume funciones suaves (puede fallar en discontinuas)
4. **Paralelización difícil**: El algoritmo es inherentemente secuencial
5. **Elección de kernel/surrogate**: Puede afectar el rendimiento significativamente

---

## Extensiones y Variantes

### Multi-Objective Bayesian Optimization

Optimizar múltiples objetivos simultáneamente:
- Maximizar accuracy
- Minimizar tiempo de entrenamiento
- Minimizar uso de memoria

**Algoritmos**: NSGA-II con GP, ParEGO

### Multi-Fidelity Bayesian Optimization

Usar evaluaciones de baja fidelidad (menos epochs, menos datos) para guiar la búsqueda:

- **Hyperband**: Asigna recursos adaptativamente
- **BOHB**: Combina Hyperband con BO

### Contextual Bayesian Optimization

Optimizar considerando contexto (ej: diferentes datasets):

$$\max_{x} f(x, c) \quad \text{donde } c \text{ es el contexto}$$

---

## Referencias Clave

1. **Snoek, J., Larochelle, H., & Adams, R. P. (2012).** "Practical Bayesian Optimization of Machine Learning Algorithms." *NeurIPS 2012*. [Paper clásico que popularizó BO en ML]

2. **Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011).** "Algorithms for Hyper-Parameter Optimization." *NeurIPS 2011*. [Introduce TPE]

3. **Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2016).** "Taking the Human Out of the Loop: A Review of Bayesian Optimization." *Proceedings of the IEEE*. [Review comprehensivo]

4. **Frazier, P. I. (2018).** "A Tutorial on Bayesian Optimization." *arXiv:1807.02811*. [Tutorial teórico detallado]

---

## Conclusión

Bayesian Optimization es una herramienta poderosa para:
- Optimización de hiperparámetros
- AutoML
- Diseño de experimentos
- Optimización de procesos industriales

**Cuándo usar BO**:
- ✅ Función objetivo costosa de evaluar (> 1 minuto por evaluación)
- ✅ Dimensionalidad baja-media (< 20 hiperparámetros)
- ✅ Budget limitado (< 500 evaluaciones)
- ✅ Necesitas interpretabilidad (incertidumbre)

**Cuándo NO usar BO**:
- ❌ Función barata (< 1 segundo) → usa Grid/Random
- ❌ Alta dimensionalidad (> 50) → usa Hyperband o evolutionary algorithms
- ❌ Budget masivo (> 10,000 evaluaciones) → Random Search puede ser suficiente

---

## Tareas Pendientes

- [ ] Crear figura: Diagrama de flujo de Bayesian Optimization
- [ ] Crear figura: GP con media y banda de incertidumbre
- [ ] Crear figura: Evolución del GP a través de iteraciones
- [ ] Crear figura: Comparación visual Grid vs Random vs Bayesian
- [ ] Crear figura: Acquisition functions (EI, PI, UCB)
- [ ] Agregar notebook de ejemplo con Optuna
- [ ] Agregar comparación empírica de tiempos de ejecución

---

**Última actualización**: Diciembre 2025
