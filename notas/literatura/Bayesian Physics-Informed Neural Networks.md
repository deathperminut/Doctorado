# Bayesian Physics-Informed Neural Networks (B-PINNs)

## Definición

Las **Physics-Informed Neural Networks (PINNs)** son modelos de deep learning en los que las **leyes físicas se integran directamente en el entrenamiento** de la red, típicamente en forma de ecuaciones diferenciales (ordinarias o parciales).

**Idea fundamental**: En lugar de entrenar solo con datos (pares entrada-salida), la red también **"respeta" las ecuaciones físicas** que gobiernan el sistema.

### Ejemplo Clásico

En vez de entrenar una red para predecir la temperatura $T(x, t)$ en un punto del espacio-tiempo a partir de datos experimentales, una PINN puede aprender la solución de la **ecuación del calor**:

$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$$

usando como restricciones la PDE y las condiciones de frontera.

**Ventaja**: Puedes entrenar con pocos datos experimentales y aún así obtener soluciones físicamente consistentes.

---

## ¿Cómo Funciona una PINN?

### Arquitectura

Una PINN tiene una **arquitectura estándar** (MLP, CNN, ResNet, etc.), pero la **función de pérdida** incluye términos adicionales que provienen de la física.

### Función de Pérdida

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{data}}}_{\text{Ajuste a datos}} + \underbrace{\lambda_{\text{phys}} \cdot \mathcal{L}_{\text{physics}}}_{\text{Consistencia física}} + \underbrace{\lambda_{\text{BC}} \cdot \mathcal{L}_{\text{boundary}}}_{\text{Condiciones de frontera}}$$

Donde:

1. **$\mathcal{L}_{\text{data}}$**: Error de predicción en puntos con datos observados
   $$\mathcal{L}_{\text{data}} = \frac{1}{N_{\text{data}}} \sum_{i=1}^{N_{\text{data}}} \|u_\theta(x_i, t_i) - u_i^{\text{obs}}\|^2$$

2. **$\mathcal{L}_{\text{physics}}$**: Error de consistencia con las ecuaciones físicas (residuo de la PDE)
   $$\mathcal{L}_{\text{physics}} = \frac{1}{N_{\text{col}}} \sum_{j=1}^{N_{\text{col}}} \left\| \mathcal{N}[u_\theta](x_j, t_j) \right\|^2$$

   donde $\mathcal{N}[\cdot]$ es el operador diferencial de la PDE.

3. **$\mathcal{L}_{\text{boundary}}$**: Error en condiciones de frontera/iniciales
   $$\mathcal{L}_{\text{boundary}} = \frac{1}{N_{\text{BC}}} \sum_{k=1}^{N_{\text{BC}}} \|u_\theta(x_k^{\text{BC}}) - g(x_k^{\text{BC}})\|^2$$

![Figura: Diagrama de arquitectura PINN]
<!-- TODO: Agregar diagrama mostrando input → NN → output con pérdidas múltiples -->

### Diferenciación Automática

**La gracia está en que las derivadas necesarias en la ecuación se calculan directamente con autodiferenciación en la red.**

Para una red $u_\theta(x, t)$, podemos calcular:

$$\frac{\partial u_\theta}{\partial x}, \quad \frac{\partial^2 u_\theta}{\partial x^2}, \quad \frac{\partial u_\theta}{\partial t}, \quad \text{etc.}$$

usando backpropagation y la regla de la cadena.

**Ejemplo**: Para la ecuación del calor:

$$\mathcal{N}[u_\theta] = \frac{\partial u_\theta}{\partial t} - \alpha \frac{\partial^2 u_\theta}{\partial x^2}$$

PyTorch/TensorFlow calculan estas derivadas automáticamente.

### Algoritmo de Entrenamiento

```
1. Inicializar red neuronal u_θ con pesos aleatorios
2. Definir puntos de colocación {(x_j, t_j)} donde evaluar la física
3. FOR cada época:
   a) Forward pass:
      - Predecir u_θ en puntos de datos
      - Predecir u_θ en puntos de colocación
      - Calcular derivadas ∂u_θ/∂x, ∂²u_θ/∂x², etc.
   b) Calcular pérdidas:
      - L_data: error con observaciones
      - L_physics: residuo de PDE
      - L_boundary: error en fronteras
   c) Backward pass:
      - Retropropagar L_total
      - Actualizar pesos θ
4. RETURN u_θ (solución aproximada)
```

![Figura: Flujo de entrenamiento PINN]
<!-- TODO: Agregar diagrama de flujo del algoritmo con flechas -->

---

## Ejemplo: Ecuación de Burgers

### Problema

Resolver la **ecuación de Burgers invíscida**:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

con condiciones:
- Inicial: $u(x, 0) = -\sin(\pi x)$
- Frontera: $u(-1, t) = u(1, t) = 0$

### Implementación en PyTorch

```python
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),  # input: (x, t)
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)   # output: u(x, t)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

def physics_loss(model, x_col, t_col, nu=0.01):
    """Calcula L_physics = ||∂u/∂t + u*∂u/∂x - ν*∂²u/∂x²||²"""
    x_col.requires_grad_(True)
    t_col.requires_grad_(True)

    u = model(x_col, t_col)

    # Primera derivada temporal
    u_t = torch.autograd.grad(u, t_col,
                              grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]

    # Primera derivada espacial
    u_x = torch.autograd.grad(u, x_col,
                              grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]

    # Segunda derivada espacial
    u_xx = torch.autograd.grad(u_x, x_col,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]

    # Residuo de la ecuación de Burgers
    residual = u_t + u * u_x - nu * u_xx

    return torch.mean(residual**2)

# Entrenamiento
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10000):
    optimizer.zero_grad()

    # Pérdida física en puntos de colocación
    x_col = torch.rand(1000, 1) * 2 - 1  # [-1, 1]
    t_col = torch.rand(1000, 1)          # [0, 1]
    loss_phys = physics_loss(model, x_col, t_col)

    # Pérdida en condición inicial
    x_ic = torch.rand(100, 1) * 2 - 1
    t_ic = torch.zeros(100, 1)
    u_pred_ic = model(x_ic, t_ic)
    u_true_ic = -torch.sin(np.pi * x_ic)
    loss_ic = torch.mean((u_pred_ic - u_true_ic)**2)

    # Pérdida total
    loss = loss_phys + 10 * loss_ic

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
```

---

## Problemas y Limitaciones de PINNs

### 1. Desvanecimiento del Gradiente

**Problema**: Cuando la PDE tiene derivadas de alto orden, el cálculo automático de derivadas puede dar **gradientes muy pequeños o inestables**.

**Ejemplo**: Para ecuaciones tipo:

$$\frac{\partial^4 u}{\partial x^4} + \text{términos de orden bajo} = 0$$

Calcular $\frac{\partial^4 u_\theta}{\partial x^4}$ requiere 4 aplicaciones sucesivas de autodiferenciación → gradientes se desvanecen.

**Soluciones parciales**:
- Normalizar los términos de la pérdida
- Usar adaptive activation functions
- Gradient clipping
- Arquitecturas especializadas (ResNets, Highway Networks)

### 2. Fenómenos Localizados (Ondas de Choque, Picos)

**Problema**: Muchas PDEs tienen soluciones con **picos/localizaciones muy bruscas**. Las PINNs tienden a suavizar todo → les cuesta capturar fenómenos localizados.

**Ejemplo**: Ecuación de Burgers con $\nu$ pequeño → forma ondas de choque (discontinuidades).

**Soluciones**:
- **Adaptive sampling**: Concentrar puntos de colocación en regiones de alta variación
- **Multi-scale architectures**: Capturar diferentes escalas con subredes
- **Fourier features**: Usar embedding de Fourier para captar alta frecuencia

### 3. Tiempo de Entrenamiento

**Problema**: A veces es **más caro entrenar la PINN que simplemente simular** el sistema con métodos numéricos tradicionales (FEM, FDM).

**Cuándo vale la pena**:
- ✅ Necesitas resolver el problema para **muchas configuraciones** diferentes
- ✅ Quieres un **surrogate diferenciable** para optimización
- ✅ Los datos son escasos y quieres **interpolar físicamente**
- ❌ Solo necesitas una simulación → usa solver tradicional

### 4. Sensibilidad a Puntos de Colocación

**Problema**: Si escoges mal los puntos $\{(x_j, t_j)\}$ donde evalúas la física, el modelo puede parecer que aprendió pero **falla en regiones no muestreadas**.

**Estrategias**:
- **Uniform sampling**: Simple pero puede ser ineficiente
- **Latin Hypercube Sampling (LHS)**: Mejor cobertura
- **Residual-based adaptive sampling**: Añadir puntos donde el residuo es alto
- **Quasi-random sequences**: Sobol, Halton

### 5. Generalización Fuera de Dominio

**Problema**: Una PINN entrenada en cierto rango de parámetros/tiempo puede **fallar rotundamente fuera de él**, aunque la ecuación física siga siendo válida.

**Ejemplo**: PINN entrenada para $t \in [0, 1]$ puede diverger para $t > 1$.

**Soluciones**:
- **Domain decomposition**: Entrenar múltiples PINNs para diferentes subdominios
- **Transfer learning**: Fine-tuning en nuevos rangos
- **Meta-learning**: Entrenar para generalizar a nuevos parámetros

### 6. Ambigüedades Físicas

**Problema**: Si el problema es **mal planteado** (múltiples soluciones posibles), la PINN no puede decidir → solo dará una "solución plausible".

**Ejemplo**: Problemas inversos sin regularización suficiente.

---

## PINNs + Bayesian Inference

### Motivación

#### **PINNs Solas** → Dan una solución puntual

- ❌ No sabes **cuánta incertidumbre** hay en la predicción
- ❌ Pueden fallar y no avisan (no sabes si la red está "segura")
- ❌ No propagas incertidumbre en parámetros físicos

#### **Métodos Bayesianos Solos** (Gaussian Processes)

- ✅ Capturan incertidumbre bien
- ❌ **Escalan mal** en problemas de alta dimensión
- ❌ Inviables para PDEs complejas (> 3D)

#### **Solución: Combinar PINNs + Bayes** ✅

- ✅ Escalabilidad de redes neuronales
- ✅ Cuantificación de incertidumbre
- ✅ Inferencia de parámetros físicos
- ✅ Detección de regiones de baja confianza

---

## Métodos de Integración

## 1. Bayesian PINNs (B-PINNs)

### Idea Central

Tratar los **pesos de la red como distribuciones probabilísticas** en vez de parámetros fijos.

Así la red ya no da una sola predicción, sino un **conjunto de soluciones probables** → con intervalos de confianza.

### Formulación Matemática

#### a) Prior sobre los pesos

Asumimos una distribución prior sobre los pesos $\theta$ de la red:

$$p(\theta) = \mathcal{N}(\theta \mid 0, \sigma_{\text{prior}}^2 I)$$

Típicamente un prior Gaussiano con varianza grande.

#### b) Likelihood

Combinamos datos observados $\mathcal{D} = \{(x_i, u_i)\}$ y consistencia física $\mathcal{F}$:

$$p(\mathcal{D}, \mathcal{F} \mid \theta) = p(\mathcal{D} \mid \theta) \cdot p(\mathcal{F} \mid \theta)$$

Donde:

- **Data likelihood**:
  $$p(\mathcal{D} \mid \theta) = \prod_{i=1}^{N_{\text{data}}} \mathcal{N}(u_i \mid u_\theta(x_i), \sigma_{\text{data}}^2)$$

- **Physics likelihood**:
  $$p(\mathcal{F} \mid \theta) = \prod_{j=1}^{N_{\text{col}}} \mathcal{N}(0 \mid \mathcal{N}[u_\theta](x_j), \sigma_{\text{phys}}^2)$$

  El residuo de la PDE debe ser cercano a 0.

#### c) Posterior sobre los pesos

Por teorema de Bayes:

$$p(\theta \mid \mathcal{D}, \mathcal{F}) = \frac{p(\mathcal{D}, \mathcal{F} \mid \theta) p(\theta)}{p(\mathcal{D}, \mathcal{F})}$$

**Problema**: Esta integral es **intratable** analíticamente.

### Métodos de Inferencia

#### **Variational Inference (VI)**

Aproximamos $p(\theta \mid \mathcal{D}, \mathcal{F})$ con una familia simple $q_\phi(\theta)$ (ej: Gaussiana):

$$q_\phi(\theta) = \mathcal{N}(\theta \mid \mu_\phi, \Sigma_\phi)$$

Minimizamos la **divergencia KL**:

$$\min_\phi \text{KL}(q_\phi(\theta) \| p(\theta \mid \mathcal{D}, \mathcal{F}))$$

Equivalente a maximizar el **ELBO** (Evidence Lower Bound):

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D}, \mathcal{F} \mid \theta)] - \text{KL}(q_\phi(\theta) \| p(\theta))$$

**Implementación**:
- Reparameterization trick: $\theta = \mu_\phi + \Sigma_\phi^{1/2} \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
- Backpropagation a través de muestras estocásticas

#### **Hamiltonian Monte Carlo (HMC)**

Muestrea directamente de $p(\theta \mid \mathcal{D}, \mathcal{F})$ usando dinámica Hamiltoniana.

**Ventajas**: Más preciso, sin sesgo de aproximación

**Desventajas**: Mucho más costoso computacionalmente

#### **Dropout como Aproximación Bayesiana**

Usar **Monte Carlo Dropout** como proxy de incertidumbre:

```python
def predict_with_uncertainty(model, x, n_samples=100):
    model.train()  # Activar dropout
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            y_pred = model(x)
            predictions.append(y_pred)

    predictions = torch.stack(predictions)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    return mean, std
```

### Ventajas de B-PINNs

- ✅ Incertidumbre bien calibrada
- ✅ Evita overconfidence típico de PINNs deterministas
- ✅ Útil si los datos son escasos o ruidosos
- ✅ Detecta regiones donde el modelo no es confiable

### Limitaciones

- ❌ Entrenar es **mucho más costoso** → cada paso requiere muestrear distribuciones
- ❌ Escala mal en redes grandes (millones de parámetros)
- ❌ Requiere tuning de hiperparámetros adicionales ($\sigma_{\text{prior}}$, $\sigma_{\text{data}}$, etc.)

![Figura: Comparación PINN vs B-PINN]
<!-- TODO: Agregar figura mostrando predicción con bandas de incertidumbre -->

---

## 2. Physics-Informed Gaussian Processes (PI-GPs)

### Idea Central

Un **Gaussian Process (GP)** es un modelo Bayesiano no paramétrico que ya da incertidumbre de forma natural.

Se puede **diseñar el kernel** del GP para que cumpla las ecuaciones físicas.

### Formulación Matemática

#### Problema a Resolver

$$\mathcal{N}[u](x) = f(x), \quad x \in \Omega$$

con condiciones de frontera.

#### GP Standard

Modelamos $u(x) \sim \text{GP}(m(x), k(x, x'))$ donde:
- $m(x)$: función media (usualmente 0)
- $k(x, x')$: función kernel (RBF, Matérn, etc.)

#### PI-GP: Kernel Informado por la Física

En vez de elegir un kernel arbitrario, lo **diseñamos para que sus derivadas cumplan la PDE**.

**Ejemplo**: Para $\nabla^2 u = f$, queremos un kernel $k$ tal que:

$$\nabla_{x}^2 k(x, x') = k_f(x, x')$$

donde $k_f$ es el kernel del término fuente $f$.

#### Construcción del Kernel

**Método 1: Green's Function**

Si $G(x, x')$ es la función de Green del operador $\mathcal{N}$:

$$\mathcal{N}[G(x, x')] = \delta(x - x')$$

Entonces usamos:

$$k_{\text{phys}}(x, x') = \int_\Omega G(x, z) k_0(z, z') G(x', z') \, dz \, dz'$$

**Método 2: Convolution Gaussian Process**

Representamos el campo como:

$$u(x) = \int_\Omega g(x - z) w(z) \, dz$$

donde $w(z) \sim \text{GP}(0, k_w)$ y $g$ es un kernel de convolución elegido para satisfacer la física.

### Predicción con PI-GP

Dados datos $\mathcal{D} = \{(x_i, u_i)\}$ y puntos de colocación $\{x_j^{\text{col}}\}$:

1. **Construir matriz de covarianza conjunta** que incluye:
   - Covarianza entre observaciones
   - Covarianza entre puntos de física (residuos)

2. **Condicionar el GP**:
   $$p(u(x^*) \mid \mathcal{D}, \mathcal{F}) = \mathcal{N}(\mu^*, \sigma^{*2})$$

3. **Obtener media y varianza** en puntos de interés $x^*$

### Ventajas de PI-GPs

- ✅ Incertidumbre **muy bien fundamentada** (teoría de GPs sólida)
- ✅ En problemas pequeños (1D, 2D) funcionan excelente
- ✅ No requieren entrenamiento iterativo (solución analítica)
- ✅ Principled way de incorporar física

### Limitaciones

- ❌ **NO ESCALAN** → En datasets grandes o dimensiones altas son inviables
  - Complejidad: $O(N^3)$ donde $N$ = número de puntos
- ❌ Difícil diseñar kernels para PDEs complejas
- ❌ Requiere conocimiento profundo de la ecuación (Green's function, etc.)

### Cuándo Usar PI-GPs

- ✅ Problemas 1D o 2D con pocos datos (< 1000 puntos)
- ✅ Necesitas incertidumbre muy precisa
- ✅ Tienes acceso a la función de Green analítica
- ❌ Problemas 3D+ o PDEs no lineales → usa B-PINNs

---

## 3. PINNs + Bayesian Inference de Parámetros

### Problema Inverso

Tienes un **modelo físico con parámetros desconocidos**:

$$\mathcal{N}[u](x; \boldsymbol{\theta}) = 0$$

donde $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_p)$ son parámetros físicos desconocidos (ej: difusividad térmica, viscosidad, constantes del material).

**Datos**: Observaciones $\mathcal{D} = \{(x_i, u_i^{\text{obs}})\}$ que pueden ser simuladas o experimentales.

**Objetivo**: Inferir la distribución posterior $p(\boldsymbol{\theta} \mid \mathcal{D})$.

### Enfoque: PINN como Solver + Bayesian Inference

#### Paso 1: Entrenar PINN como Solver Rápido

Entrenar una PINN $u_{\theta_{\text{NN}}}(x; \boldsymbol{\theta}_{\text{phys}})$ **condicionada en los parámetros físicos** $\boldsymbol{\theta}_{\text{phys}}$.

**Función de pérdida**:

$$\mathcal{L} = \underbrace{\frac{1}{N_{\text{data}}} \sum_i \|u_{\theta_{\text{NN}}}(x_i; \boldsymbol{\theta}_{\text{phys}}) - u_i^{\text{obs}}\|^2}_{\text{Error contra datos}} + \underbrace{\lambda \cdot \frac{1}{N_{\text{col}}} \sum_j \|\mathcal{N}[u_{\theta_{\text{NN}}}](x_j; \boldsymbol{\theta}_{\text{phys}})\|^2}_{\text{Residuo físico}}$$

Aquí $\boldsymbol{\theta}_{\text{phys}}$ se trata como **entrada adicional** a la red.

**Arquitectura**:
```
Input: [x, t, θ_phys] → MLP → Output: u(x, t; θ_phys)
```

#### Paso 2: Definir Modelo Bayesiano

**Prior sobre parámetros**:

$$p(\boldsymbol{\theta}_{\text{phys}}) = \prod_{k=1}^p p(\theta_k)$$

Por ejemplo, priors uniformes o log-normales según conocimiento del dominio.

**Likelihood**:

$$p(\mathcal{D} \mid \boldsymbol{\theta}_{\text{phys}}) = \prod_{i=1}^{N_{\text{data}}} \mathcal{N}\left(u_i^{\text{obs}} \mid u_{\theta_{\text{NN}}}(x_i; \boldsymbol{\theta}_{\text{phys}}), \sigma_{\text{noise}}^2\right)$$

La PINN actúa como un **modelo forward diferenciable**.

**Posterior**:

$$p(\boldsymbol{\theta}_{\text{phys}} \mid \mathcal{D}) \propto p(\mathcal{D} \mid \boldsymbol{\theta}_{\text{phys}}) \cdot p(\boldsymbol{\theta}_{\text{phys}})$$

#### Paso 3: Inferencia

**Métodos**:

1. **MCMC (Markov Chain Monte Carlo)**:
   - Metropolis-Hastings
   - Hamiltonian Monte Carlo (HMC)
   - No-U-Turn Sampler (NUTS)

2. **Variational Inference**:
   - Aproximar $p(\boldsymbol{\theta}_{\text{phys}} \mid \mathcal{D})$ con $q_\phi(\boldsymbol{\theta}_{\text{phys}})$
   - Más rápido pero menos preciso

3. **Bayesian Optimization**:
   - Si solo necesitas el máximo a posteriori (MAP)

### Ejemplo: Inferir Difusividad Térmica

**Ecuación**:

$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$$

donde $\alpha$ es desconocido.

**Implementación**:

```python
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

class ConditionalPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 50),  # [x, y, t, alpha]
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)   # T(x, y, t; alpha)
        )

    def forward(self, x, y, t, alpha):
        inputs = torch.cat([x, y, t, alpha], dim=1)
        return self.net(inputs)

def model(data_x, data_y, data_t, data_T_obs):
    # Prior sobre alpha
    alpha = pyro.sample("alpha", dist.Uniform(0.001, 0.1))

    # PINN forward pass
    alpha_tensor = alpha * torch.ones(len(data_x), 1)
    T_pred = pinn_model(data_x, data_y, data_t, alpha_tensor)

    # Likelihood
    sigma = pyro.sample("sigma", dist.Uniform(0.001, 0.1))
    with pyro.plate("data", len(data_x)):
        pyro.sample("obs", dist.Normal(T_pred.squeeze(), sigma), obs=data_T_obs)

# Inferencia con NUTS
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500)
mcmc.run(data_x, data_y, data_t, data_T_obs)

# Obtener posterior
posterior_samples = mcmc.get_samples()
alpha_samples = posterior_samples['alpha']

print(f"Alpha estimado: {alpha_samples.mean():.4f} ± {alpha_samples.std():.4f}")
```

### Ventajas

- ✅ Cuantifica incertidumbre en **parámetros físicos** no solo en predicciones
- ✅ Permite **calibración de modelos** con datos experimentales
- ✅ Detecta parámetros no identificables (posterior muy amplio)
- ✅ Útil para **validación de modelos** y diseño de experimentos

### Limitaciones

- ❌ Requiere entrenar una PINN condicional (más complejo)
- ❌ Inferencia puede ser costosa (especialmente MCMC)
- ❌ Sensible a la calidad del solver PINN

---

## Aplicación: Imágenes de Dominios Magnéticos → Parámetros Hamiltonianos

### Problema

**Input**: Imagen de configuración de dominios magnéticos $I \in \mathbb{R}^{H \times W}$

**Output**: Parámetros del Hamiltoniano $\boldsymbol{\theta} = (J_{\text{ex}}, K_{\text{DM}}, T, \ldots)$

**Challenge**: Problema inverso mal planteado → múltiples $\boldsymbol{\theta}$ pueden generar imágenes similares.

### Enfoque Bayesiano con PINNs

#### Paso 1: Entrenar Modelo Generador Forward

**Opción A: PINN Condicionada** (Físicamente explícita)

Entrenar una PINN que resuelve el Hamiltoniano de Heisenberg:

$$\mathcal{H} = -J_{\text{ex}} \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j - K_{\text{DM}} \sum_{\langle i,j \rangle} \mathbf{D}_{ij} \cdot (\mathbf{S}_i \times \mathbf{S}_j) + \ldots$$

- **Input**: Parámetros $\boldsymbol{\theta} = (J_{\text{ex}}, K_{\text{DM}}, T, \ldots)$
- **Output**: Campo de spins $\{\mathbf{S}_i\}$ representado como imagen

**Arquitectura**:
```
θ = [J_ex, K_DM, T, ...] → MLP/CNN → S(x, y) → render → Imagen I
```

**Función de pérdida**:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{data}}}_{\text{Ajuste a simulaciones}} + \underbrace{\lambda_1 \cdot \mathcal{L}_{\text{Hamiltonian}}}_{\text{Energía física}} + \underbrace{\lambda_2 \cdot \mathcal{L}_{\text{stability}}}_{\text{Estabilidad temporal}}$$

Donde:
- $\mathcal{L}_{\text{Hamiltonian}}$: Minimizar energía del Hamiltoniano
- $\mathcal{L}_{\text{stability}}$: Asegurar que la configuración es estable

**Opción B: CNN/U-Net Surrogate** (Más práctico)

Entrenar una red convolucional en un dataset de simulaciones:

- **Dataset**: $\{(\boldsymbol{\theta}_i, I_i)\}_{i=1}^N$ de simulaciones atomísticas (VAMPIRE, Spirit, etc.)
- **Arquitectura**: Conditional GAN o U-Net
  ```
  θ → [Embedding] → [Decoder/Generator] → Imagen I
  ```

**Ventaja**: Más rápido de entrenar, no requiere resolver PDE explícitamente

**Desventaja**: Menos físicamente interpretable, puede generar configuraciones no físicas

#### Paso 2: Modelo Bayesiano Inverso

**Prior sobre parámetros**:

$$p(\boldsymbol{\theta}) = \prod_k p(\theta_k)$$

Basado en conocimiento físico:
- $J_{\text{ex}} \sim \text{LogNormal}(\mu_{J}, \sigma_J)$ (intercambio siempre positivo)
- $K_{\text{DM}} \sim \text{Uniform}(K_{\min}, K_{\max})$
- $T \sim \text{Uniform}(0, T_{\text{Curie}})$

**Likelihood**:

$$p(I^{\text{obs}} \mid \boldsymbol{\theta}) = \mathcal{N}(I^{\text{obs}} \mid G(\boldsymbol{\theta}), \sigma_{\text{noise}}^2 \cdot \mathbf{I})$$

donde $G(\boldsymbol{\theta})$ es el modelo generador (PINN o CNN).

**Posterior**:

$$p(\boldsymbol{\theta} \mid I^{\text{obs}}) \propto p(I^{\text{obs}} \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta})$$

#### Paso 3: Inferencia

**Método 1: MCMC**

```python
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

def model(I_obs):
    # Priors
    J_ex = pyro.sample("J_ex", dist.LogNormal(0.0, 1.0))
    K_DM = pyro.sample("K_DM", dist.Uniform(-1.0, 1.0))
    T = pyro.sample("T", dist.Uniform(0.0, 300.0))

    # Forward model (PINN o CNN)
    theta = torch.tensor([J_ex, K_DM, T])
    I_pred = forward_model(theta)  # PINN o CNN

    # Likelihood
    sigma = pyro.sample("sigma", dist.Uniform(0.01, 0.1))
    with pyro.plate("pixels", I_obs.numel()):
        pyro.sample("obs", dist.Normal(I_pred.flatten(), sigma),
                    obs=I_obs.flatten())

# Inferencia
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=1000)
mcmc.run(I_observed)

# Posterior
samples = mcmc.get_samples()
J_ex_posterior = samples['J_ex']
K_DM_posterior = samples['K_DM']
T_posterior = samples['T']
```

**Método 2: Variational Inference** (más rápido)

```python
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

def guide(I_obs):
    # Variational parameters
    J_ex_loc = pyro.param("J_ex_loc", torch.tensor(0.0))
    J_ex_scale = pyro.param("J_ex_scale", torch.tensor(1.0),
                            constraint=constraints.positive)

    K_DM_loc = pyro.param("K_DM_loc", torch.tensor(0.0))
    K_DM_scale = pyro.param("K_DM_scale", torch.tensor(0.5),
                           constraint=constraints.positive)

    # Variational distributions
    pyro.sample("J_ex", dist.Normal(J_ex_loc, J_ex_scale))
    pyro.sample("K_DM", dist.Normal(K_DM_loc, K_DM_scale))
    # ... etc

# Optimización
adam = Adam({"lr": 0.01})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

for step in range(5000):
    loss = svi.step(I_observed)
    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")
```

**Método 3: Bayesian Optimization** (si solo necesitas MAP)

Ver documento de Bayesian Optimization.

#### Paso 4: Análisis de Resultados

**Visualizaciones**:

1. **Distribuciones marginales** de cada parámetro:
   ```python
   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 3, figsize=(15, 4))
   axes[0].hist(J_ex_posterior, bins=50, density=True)
   axes[0].set_xlabel('$J_{ex}$')
   axes[1].hist(K_DM_posterior, bins=50, density=True)
   axes[1].set_xlabel('$K_{DM}$')
   axes[2].hist(T_posterior, bins=50, density=True)
   axes[2].set_xlabel('$T$ (K)')
   ```

2. **Pairplot** (correlaciones entre parámetros):
   ```python
   import seaborn as sns
   import pandas as pd

   df = pd.DataFrame({
       'J_ex': J_ex_posterior.numpy(),
       'K_DM': K_DM_posterior.numpy(),
       'T': T_posterior.numpy()
   })
   sns.pairplot(df)
   ```

3. **Posterior predictive checks**:
   ```python
   # Generar imágenes con muestras del posterior
   n_samples = 100
   I_predicted = []

   for i in range(n_samples):
       theta_sample = [J_ex_posterior[i], K_DM_posterior[i], T_posterior[i]]
       I_pred = forward_model(torch.tensor(theta_sample))
       I_predicted.append(I_pred)

   I_predicted = torch.stack(I_predicted)
   I_mean = I_predicted.mean(dim=0)
   I_std = I_predicted.std(dim=0)

   # Visualizar
   fig, axes = plt.subplots(1, 3, figsize=(15, 4))
   axes[0].imshow(I_obs)
   axes[0].set_title('Observed')
   axes[1].imshow(I_mean)
   axes[1].set_title('Posterior Mean')
   axes[2].imshow(I_std)
   axes[2].set_title('Posterior Std')
   ```

### Ventajas del Enfoque Bayesiano

- ✅ **Cuantifica incertidumbre** en parámetros estimados
- ✅ **Detecta parámetros no identificables** (posterior muy amplio)
- ✅ **Propaga incertidumbre** a predicciones futuras
- ✅ **Permite comparación de modelos** (Bayes factors)
- ✅ **Robusto a ruido** en observaciones experimentales

### Desafíos

- ❌ **Computacionalmente costoso**: MCMC puede requerir miles de evaluaciones del forward model
- ❌ **Sensibilidad a priors**: Priors mal elegidos → posterior sesgado
- ❌ **Múltiples modos**: El posterior puede ser multimodal (varios $\boldsymbol{\theta}$ explican los datos)
- ❌ **Escalabilidad**: Para imágenes grandes o muchos parámetros, la inferencia es lenta

---

## Comparación de Enfoques

| Enfoque | Escalabilidad | Incertidumbre | Física Explícita | Complejidad |
|---------|--------------|---------------|------------------|-------------|
| **PINN Standard** | Alta | ❌ No | ✅ Sí | Media |
| **B-PINN (VI)** | Media | ✅ Sí (aproximada) | ✅ Sí | Alta |
| **B-PINN (HMC)** | Baja | ✅ Sí (exacta) | ✅ Sí | Muy Alta |
| **PI-GP** | Muy Baja | ✅ Sí (exacta) | ✅ Sí (kernel) | Media |
| **PINN + Bayesian Params** | Media-Alta | ✅ Sí (parámetros) | ✅ Sí | Alta |
| **CNN Surrogate + Bayes** | Alta | ✅ Sí (parámetros) | ❌ No | Media-Alta |

**Recomendaciones**:

- **Problema forward, < 3D, < 1000 puntos** → PI-GP
- **Problema forward, alta dimensión, necesitas incertidumbre** → B-PINN (VI)
- **Problema inverso, pocos parámetros (< 10)** → PINN + Bayesian inference (MCMC)
- **Problema inverso, muchos parámetros, prioridad velocidad** → CNN Surrogate + Bayesian Optimization
- **Solo necesitas predicción rápida, no incertidumbre** → PINN Standard

---

## Herramientas y Librerías

### Para PINNs

1. **DeepXDE** ⭐ [RECOMENDADO]
   - URL: https://deepxde.readthedocs.io/
   - Backend: TensorFlow, PyTorch, JAX
   - Incluye ejemplos de muchas PDEs

   ```python
   import deepxde as dde

   def pde(x, y):
       dy_t = dde.grad.jacobian(y, x, i=0, j=1)
       dy_xx = dde.grad.hessian(y, x, i=0, j=0)
       return dy_t - dy_xx

   geom = dde.geometry.Interval(-1, 1)
   timedomain = dde.geometry.TimeDomain(0, 1)
   geomtime = dde.geometry.GeometryXTime(geom, timedomain)

   data = dde.data.TimePDE(geomtime, pde, [], num_domain=2540,
                           num_boundary=80, num_initial=160)
   net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
   model = dde.Model(data, net)
   model.compile("adam", lr=1e-3)
   model.train(epochs=15000)
   ```

2. **NeuralPDE.jl** (Julia)
   - URL: https://neuralpde.sciml.ai/
   - Integración con SciML ecosystem
   - Muy eficiente

3. **PyDEns** (Python)
   - URL: https://github.com/analysiscenter/pydens
   - Interfaz simple

### Para Bayesian Inference

1. **Pyro** ⭐ [RECOMENDADO]
   - URL: https://pyro.ai/
   - Basado en PyTorch
   - Soporta VI y MCMC (NUTS, HMC)

2. **PyMC**
   - URL: https://www.pymc.io/
   - API muy limpia
   - Excelente para modelos probabilísticos

3. **TensorFlow Probability**
   - URL: https://www.tensorflow.org/probability
   - Para usuarios de TensorFlow

### Para Gaussian Processes

1. **GPyTorch**
   - URL: https://gpytorch.ai/
   - GP escalables en PyTorch

2. **GPflow**
   - URL: https://gpflow.github.io/GPflow/
   - Basado en TensorFlow

---

## Referencias Clave

### PINNs Fundamentales

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707. [Paper fundacional]

2. **Karniadakis, G. E., et al. (2021).** "Physics-informed machine learning." *Nature Reviews Physics*, 3(6), 422-440. [Review comprehensivo]

### Bayesian PINNs

3. **Yang, L., Meng, X., & Karniadakis, G. E. (2021).** "B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data." *Journal of Computational Physics*, 425, 109913. [B-PINNs original]

4. **Psaros, A. F., Kawaguchi, K., & Karniadakis, G. E. (2022).** "Meta-learning PINN loss functions." *Journal of Computational Physics*, 458, 111121.

### Physics-Informed GPs

5. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017).** "Inferring solutions of differential equations using noisy multi-fidelity data." *Journal of Computational Physics*, 335, 736-746.

6. **Gulian, M., Raissi, M., Perdikaris, P., & Karniadakis, G. (2019).** "Multistep Neural Networks for Data-driven Discovery of Nonlinear Dynamical Systems." *arXiv:1801.01236*.

### Aplicaciones

7. **Cai, S., et al. (2021).** "Physics-Informed Neural Networks for Heat Transfer Problems." *Journal of Heat Transfer*, 143(6).

8. **Meng, X., & Karniadakis, G. E. (2020).** "A composite neural network that learns from multi-fidelity data: Application to function approximation and inverse PDE problems." *Journal of Computational Physics*, 401, 109020.

---

## Conclusión

Las **Bayesian Physics-Informed Neural Networks** combinan lo mejor de dos mundos:

- ✅ **Escalabilidad** de deep learning
- ✅ **Consistencia física** de métodos basados en ecuaciones
- ✅ **Cuantificación de incertidumbre** de métodos Bayesianos

**Cuándo usar B-PINNs**:
- ✅ Datos escasos o ruidosos
- ✅ Necesitas cuantificar incertidumbre
- ✅ Problema inverso mal planteado
- ✅ Validación de modelos físicos

**Cuándo usar PINNs Standard**:
- ✅ Muchos datos limpios
- ✅ Solo necesitas predicción rápida
- ✅ Recursos computacionales limitados

**Cuándo usar PI-GPs**:
- ✅ Problemas pequeños (1D-2D)
- ✅ Incertidumbre muy precisa crítica
- ✅ Pocos datos (< 1000 puntos)

**Para tu aplicación (dominios magnéticos)**:
- **Recomendación**: PINN Condicionada + Bayesian Inference (MCMC)
- **Alternativa rápida**: CNN Surrogate + Variational Inference
- **Si necesitas máxima precisión**: B-PINN con HMC (más costoso)

---

## Tareas Pendientes

- [ ] Crear figura: Arquitectura PINN con múltiples pérdidas
- [ ] Crear figura: Flujo de entrenamiento PINN
- [ ] Crear figura: Comparación PINN vs B-PINN (con bandas de incertidumbre)
- [ ] Crear figura: Diagrama de flujo aplicación a dominios magnéticos
- [ ] Implementar ejemplo completo: B-PINN para ecuación del calor
- [ ] Implementar ejemplo: Inferencia Bayesiana de parámetros magnéticos
- [ ] Benchmark: Comparar tiempos de ejecución MCMC vs VI

---

**Última actualización**: Diciembre 2025
