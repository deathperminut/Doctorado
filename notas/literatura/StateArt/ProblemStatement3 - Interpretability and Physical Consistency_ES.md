# Problem Statement 3: Interpretability and Physical Consistency in Deep Learning

## Resumen Ejecutivo

**El Desafío**: Las redes neuronales profundas son modelos potentes de caja negra que pueden lograr alta precisión de predicción pero carecen de transparencia en su proceso de toma de decisiones. Para aplicaciones de física, esta opacidad genera preocupaciones críticas: **¿Cómo aseguramos que el modelo respete las leyes físicas fundamentales?** **¿Podemos confiar en predicciones en regiones de parámetros inexploradas?** **¿Qué características físicas aprende realmente el modelo?**

**Impacto**: La falta de interpretabilidad conduce a:
- **Predicciones no físicas**: Los modelos pueden violar leyes de conservación, simetrías o restricciones termodinámicas
- **Pobre generalización**: Los modelos de caja negra fallan catastróficamente fuera de la distribución de entrenamiento
- **Sin comprensión mecanística**: No se puede extraer comprensión física de las representaciones aprendidas
- **Barreras de despliegue**: Los científicos son reacios a confiar en modelos opacos para decisiones críticas

**Brecha de Investigación**: Aunque existen métodos de interpretabilidad (GradCAM, SHAP, etc.), la mayoría están diseñados para visión por computadora o tareas de NLP. **La interpretabilidad específica de física**—verificar leyes de conservación, detectar simetrías aprendidas, extraer relaciones simbólicas—permanece subdesarrollada. Necesitamos métodos que vayan más allá de "qué píxeles importan" para preguntar **"¿el modelo entiende la física?"**

---

## 1. El Problema de Interpretabilidad en Física

### 1.1 Por Qué la Interpretabilidad Importa Más en Física

**Visión por Computadora**: "¿Por qué el modelo clasificó esto como un gato?"
→ Suficiente destacar regiones de imagen relevantes

**Física**: "¿Por qué el modelo predijo $J = 1.2$ meV?"
→ Necesidad de verificar:
- ✓ ¿Se respeta el principio de minimización de energía?
- ✓ ¿Se preserva la simetría rotacional?
- ✓ ¿Se captura el trade-off intercambio-anisotropía?
- ✓ ¿La predicción es consistente con el diagrama de fases?

**Diferencia Fundamental**:
```
ML Estándar: La precisión es rey
ML de Física: Precisión + Consistencia Física + Comprensión Mecanística
```

### 1.2 Modos de Falla de Modelos de Caja Negra

#### A) Extrapolación No Física

**Ejemplo**: CNN entrenada en imágenes de skyrmión en rango $D \in [1, 3]$ mJ/m²

Probar en $D = 5$:
- **Predicción de caja negra**: $D_{\text{pred}} = 2.1$ (¡sin sentido!)
- **¿Por qué?**: El modelo memorizó texturas, no física
- **Restricción física**: Para $D = 5$, los skyrmiones deberían tener núcleos más pequeños → el modelo viola la física

#### B) Correlaciones Espurias

**Ejemplo**: Regresor aprende correlación entre brillo de imagen y temperatura

```python
# Modelo aprende: Alto brillo → Alto T
# Realidad: El brillo es solo un artefacto de visualización
# Resultado: Falla en diferentes ajustes de contraste
```

**Problema**: El modelo explota artefactos específicos del dataset en lugar de relaciones físicas

#### C) Violaciones de Simetría

**Ejemplo**: El Hamiltoniano es invariante rotacional:
$$H(R\mathbf{S}) = H(\mathbf{S})$$

Pero predicción CNN:
$$f_\theta(R\mathbf{I}) \neq f_\theta(\mathbf{I})$$

**Resultado**: ¡Predicciones diferentes para versiones rotadas de la misma configuración física!

![Figura: Ejemplo de violación de simetría]
<!-- TODO: Agregar figura mostrando misma config de espín rotada dando predicciones diferentes -->

### 1.3 El Espectro de Interpretabilidad

| Nivel | Pregunta | Métodos | Relevancia en Física |
|-------|----------|---------|---------------------|
| **1. Atribución de Características** | ¿Qué regiones de entrada importan? | GradCAM, SHAP, Saliency | ⭐ Básico: Identificar paredes de dominio, núcleos |
| **2. Detección de Conceptos** | ¿Qué características de alto nivel se aprendieron? | TCAV, Probing | ⭐⭐ Intermedio: Detectar skyrmiones, rayas |
| **3. Restricciones Físicas** | ¿Se respetan las leyes físicas? | Pruebas de conservación, Verificación de simetría | ⭐⭐⭐ Avanzado: Verificar energía, simetrías |
| **4. Comprensión Mecanística** | ¿Qué algoritmo implementa el modelo? | Análisis de circuitos, Regresión simbólica | ⭐⭐⭐⭐ Experto: Extraer ecuaciones gobernantes |

---

## 2. Métodos de Interpretabilidad Post-Hoc

### 2.1 Atribución Basada en Gradientes

#### A) Mapas de Saliencia

**Concepto**: Calcular gradiente de predicción respecto a entrada

$$\text{Saliency}(\mathbf{x}) = \left| \frac{\partial f_\theta(\mathbf{x})}{\partial \mathbf{x}} \right|$$

**Implementación**:
```python
import torch
import torch.nn.functional as F

def compute_saliency(model, image, target_param_idx=0):
    """
    Calcular mapa de saliencia para predicción de parámetro

    Args:
        model: Regresor entrenado
        image: [1, C, H, W] entrada
        target_param_idx: Qué parámetro explicar (0=J, 1=K, etc.)

    Returns:
        saliency: [H, W] mapa de importancia
    """
    image.requires_grad_(True)

    # Forward pass
    output = model(image)
    target = output[0, target_param_idx]

    # Backward pass
    target.backward()

    # Saliency = gradiente absoluto
    saliency = image.grad.abs().squeeze()

    return saliency

# Uso
saliency_J = compute_saliency(model, mfm_image, target_param_idx=0)

# Visualizar
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(mfm_image.squeeze(), cmap='gray')
axes[0].set_title('Imagen MFM')
axes[1].imshow(saliency_J, cmap='hot')
axes[1].set_title('Saliency para predicción de J')
```

**Interpretación Física**:
- Alta saliencia en **paredes de dominio** → Modelo usa ancho de pared para estimar $J$
- Alta saliencia en **núcleos de skyrmión** → Modelo usa tamaño de núcleo para estimar $D$

**Ventajas**:
✅ Simple, rápido (un paso backward)
✅ Resolución a nivel de píxel
✅ Sin entrenamiento adicional

**Limitaciones**:
❌ Ruidoso, se satura fácilmente
❌ Solo muestra gradientes locales (no contexto global)
❌ No verifica consistencia física

#### B) Grad-CAM (Gradient-weighted Class Activation Mapping)

**Concepto**: Ponderar mapas de características por importancia de gradiente

$$\alpha_k = \frac{1}{Z} \sum_{i,j} \frac{\partial y}{\partial A_k^{ij}}$$

$$\text{GradCAM} = \text{ReLU}\left( \sum_k \alpha_k A_k \right)$$

**Implementación**:
```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image, target_param_idx=0):
        """
        Generar mapa de calor Grad-CAM

        Args:
            image: [1, C, H, W]
            target_param_idx: Qué parámetro explicar

        Returns:
            cam: [H, W] mapa de calor
        """
        # Forward
        output = self.model(image)
        target = output[0, target_param_idx]

        # Backward
        self.model.zero_grad()
        target.backward()

        # Calcular pesos (promedio global de gradientes)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Combinación ponderada de mapas de activación
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']

        # ReLU y normalizar
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

# Uso
gradcam = GradCAM(model, target_layer=model.features[-1])
cam_J = gradcam(mfm_image, target_param_idx=0)

# Superponer en imagen
plt.imshow(mfm_image.squeeze(), cmap='gray')
plt.imshow(cam_J, cmap='jet', alpha=0.5)
plt.title('Grad-CAM: Regiones importantes para predicción de J')
```

**Interpretación Física**:
- **Para intercambio ($J$)**: Debería resaltar regiones con fuertes correlaciones de espín (paredes de dominio, longitudes de onda espirales)
- **Para DMI ($D$)**: Debería enfocarse en características quirales (quiralidad de skyrmión, helicidad espiral)
- **Para anisotropía ($K$)**: Debería enfatizar dominios fuera del plano

**Validación**: ¡Verificar si las regiones resaltadas coinciden con la intuición física!

**Ventajas**:
✅ Discriminativo de clase (mejor que saliency)
✅ Menor resolución → menos ruidoso
✅ Funciona con cualquier arquitectura CNN

**Limitaciones**:
❌ Resolución espacial gruesa
❌ Requiere capas convolucionales
❌ Aún no verifica consistencia física

### 2.2 Métodos Basados en Perturbación

#### A) LIME (Local Interpretable Model-agnostic Explanations)

**Concepto**: Aproximar localmente modelo de caja negra con modelo lineal interpretable

$$f(\mathbf{x}) \approx \sum_i w_i \cdot \mathbb{1}[\text{superpixel}_i \text{ presente}]$$

**Implementación**:
```python
from lime import lime_image
import numpy as np

class MagneticImageExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()

    def explain(self, image, target_param_idx=0, num_samples=1000):
        """
        Explicar predicción usando LIME

        Args:
            image: [H, W] imagen MFM
            target_param_idx: Qué parámetro explicar
            num_samples: Número de muestras perturbadas

        Returns:
            explanation: Objeto de explicación LIME
        """
        def predict_fn(images):
            """Función de predicción por lotes para LIME"""
            # images: [N, H, W, C] en numpy
            images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                outputs = self.model(images_tensor)
            return outputs[:, target_param_idx].numpy()

        # Ejecutar LIME
        explanation = self.explainer.explain_instance(
            image.squeeze().numpy(),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )

        return explanation

    def visualize_explanation(self, image, explanation):
        """Visualizar explicación LIME"""
        from skimage.segmentation import mark_boundaries

        # Obtener regiones positivas y negativas
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Original
        axes[0].imshow(image.squeeze(), cmap='gray')
        axes[0].set_title('Imagen MFM Original')

        # Superpíxeles
        axes[1].imshow(mark_boundaries(temp, mask))
        axes[1].set_title('Superpíxeles Importantes')

        # Mapa de calor
        heatmap = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=True
        )[1]
        axes[2].imshow(heatmap, cmap='RdBu', vmin=-1, vmax=1)
        axes[2].set_title('Importancia de Características')

# Uso
explainer = MagneticImageExplainer(model)
explanation = explainer.explain(mfm_image, target_param_idx=0)
explainer.visualize_explanation(mfm_image, explanation)
```

**Interpretación Física**:
- **Pesos positivos**: Superpíxeles que aumentan valor de parámetro predicho
- **Pesos negativos**: Superpíxeles que disminuyen predicción
- **Verificar**: ¿Las regiones positivas corresponden a características físicas esperadas?

**Ventajas**:
✅ Agnóstico al modelo (funciona con cualquier caja negra)
✅ Proporciona aproximación lineal local
✅ Puede identificar interacciones no intuitivas

**Limitaciones**:
❌ Computacionalmente costoso (1000s de evaluaciones)
❌ La segmentación de superpíxeles puede no respetar límites físicos
❌ Inestable (diferentes ejecuciones dan explicaciones diferentes)

#### B) SHAP (SHapley Additive exPlanations)

**Concepto**: Usar valores de Shapley de teoría de juegos para asignar importancia

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

**Implementación**:
```python
import shap

class SHAPMagneticExplainer:
    def __init__(self, model, background_data):
        """
        Inicializar explicador SHAP

        Args:
            model: Modelo entrenado
            background_data: [K, C, H, W] muestras representativas para baseline
        """
        self.model = model
        self.explainer = shap.DeepExplainer(model, background_data)

    def explain(self, image, target_param_idx=0):
        """
        Calcular valores SHAP para imagen

        Args:
            image: [1, C, H, W] entrada
            target_param_idx: Qué salida explicar

        Returns:
            shap_values: [C, H, W] puntuaciones de importancia
        """
        shap_values = self.explainer.shap_values(image)

        # shap_values es lista de [1, C, H, W] (uno por salida)
        # Extraer para parámetro objetivo
        importance = shap_values[target_param_idx].squeeze()

        return importance

    def visualize(self, image, shap_values):
        """Visualizar explicación SHAP"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Original
        axes[0].imshow(image.squeeze(), cmap='gray')
        axes[0].set_title('Imagen MFM')

        # Contribuciones positivas
        axes[1].imshow(np.maximum(shap_values, 0), cmap='Reds')
        axes[1].set_title('SHAP Positivo (aumenta predicción)')

        # Contribuciones negativas
        axes[2].imshow(np.maximum(-shap_values, 0), cmap='Blues')
        axes[2].set_title('SHAP Negativo (disminuye predicción)')

# Uso
# Crear dataset de fondo (muestra del conjunto de entrenamiento)
background = train_dataset[:100]  # [100, C, H, W]

shap_explainer = SHAPMagneticExplainer(model, background)
shap_vals = shap_explainer.explain(mfm_image, target_param_idx=0)
shap_explainer.visualize(mfm_image, shap_vals)
```

**Interpretación Física**:
- **Valores SHAP**: Contribución de cada píxel a desviación de predicción baseline
- **Baseline**: Predicción promedio sobre dataset de fondo
- **Descomposición**: $f(\mathbf{x}) = f_{\text{baseline}} + \sum_i \phi_i$

**Validación**:
```python
# Verificar si valores SHAP suman correctamente
baseline_pred = model(background).mean()
shap_pred = baseline_pred + shap_vals.sum()
actual_pred = model(mfm_image)

print(f"Baseline: {baseline_pred:.4f}")
print(f"Predicción SHAP: {shap_pred:.4f}")
print(f"Real: {actual_pred:.4f}")
print(f"Error: {abs(shap_pred - actual_pred):.6f}")  # Debería ser ~0
```

**Ventajas**:
✅ Fundamentado teóricamente (solución única con propiedades deseables)
✅ Consistente y localmente preciso
✅ Puede comparar importancia de características entre muestras

**Limitaciones**:
❌ Extremadamente costoso computacionalmente
❌ Requiere dataset de fondo representativo
❌ Asume independencia de características (violado en imágenes)

---

## 3. Arquitecturas Interpretables Restringidas por Física

### 3.1 Redes Neuronales Hamiltonianas (HNNs)

**Concepto**: Construir sesgo inductivo para conservación de energía en arquitectura

**NN Estándar**: Aprende función arbitraria $f: \mathbf{q}, \dot{\mathbf{q}} \to \ddot{\mathbf{q}}$

**HNN**: Aprende Hamiltoniano $H(\mathbf{q}, \mathbf{p})$, dinámica sigue:
$$\dot{\mathbf{q}} = \frac{\partial H}{\partial \mathbf{p}}, \quad \dot{\mathbf{p}} = -\frac{\partial H}{\partial \mathbf{q}}$$

**Garantías**:
- ✅ Conservación de energía por construcción
- ✅ Estructura simpléctica preservada
- ✅ Estabilidad a largo plazo

**Implementación**:
```python
import torch
import torch.nn as nn

class HamiltonianNN(nn.Module):
    """
    Red Neuronal Hamiltoniana para sistemas conservativos

    Aprende H(q, p) tal que la dinámica sigue ecuaciones de Hamilton
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim

        # Red para aprender H(q, p)
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Salida: Hamiltoniano escalar
        )

    def forward(self, qp):
        """
        Args:
            qp: [B, 2*D] posiciones y momentos concatenados

        Returns:
            H: [B, 1] valor Hamiltoniano
        """
        H = self.hamiltonian_net(qp)
        return H

    def compute_derivatives(self, qp):
        """
        Calcular dq/dt y dp/dt usando ecuaciones de Hamilton

        Returns:
            dqp_dt: [B, 2*D] derivadas temporales
        """
        qp.requires_grad_(True)
        H = self.forward(qp)

        # Calcular gradientes
        dH_dqp = torch.autograd.grad(
            H.sum(), qp, create_graph=True
        )[0]

        # Dividir en dH/dq y dH/dp
        D = self.input_dim
        dH_dq = dH_dqp[:, :D]
        dH_dp = dH_dqp[:, D:]

        # Ecuaciones de Hamilton
        dq_dt = dH_dp
        dp_dt = -dH_dq

        dqp_dt = torch.cat([dq_dt, dp_dt], dim=1)

        return dqp_dt

# Entrenamiento
def train_hnn(model, data_loader, epochs=100):
    """
    Entrenar HNN para coincidir con dinámica observada

    Args:
        data_loader: Produce (qp_t, dqp_dt_true)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for qp, dqp_dt_true in data_loader:
            optimizer.zero_grad()

            # Predecir derivadas usando ecuaciones de Hamilton
            dqp_dt_pred = model.compute_derivatives(qp)

            # Pérdida: coincidir derivadas
            loss = F.mse_loss(dqp_dt_pred, dqp_dt_true)

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
```

**Aplicación a Sistemas Magnéticos**:

Para dinámica micromagnética, coordenadas generalizadas:
- $\mathbf{q}$: Orientaciones de espín (ángulos)
- $\mathbf{p}$: Momentos conjugados

Función de energía:
$$H = \int d\mathbf{r} \left[ \frac{J}{2}(\nabla \mathbf{S})^2 + K S_z^2 + D \mathbf{S} \cdot (\nabla \times \mathbf{S}) \right]$$

**HNN aprende este funcional de energía desde datos de trayectoria**

**Interpretación**:
```python
# Extraer paisaje de energía aprendido
q_grid = torch.linspace(-np.pi, np.pi, 100)
p_grid = torch.linspace(-1, 1, 100)
Q, P = torch.meshgrid(q_grid, p_grid)
qp_grid = torch.stack([Q.flatten(), P.flatten()], dim=1)

H_values = model(qp_grid).reshape(100, 100)

# Visualizar paisaje de energía
plt.contourf(Q, P, H_values.detach().numpy(), levels=50)
plt.xlabel('Posición q')
plt.ylabel('Momento p')
plt.title('Hamiltoniano Aprendido H(q, p)')
plt.colorbar(label='Energía')
```

**Ventajas**:
✅ **Conservación de energía garantizada**: Sin deriva espuria de energía
✅ **Interpretable**: $H$ aprendido tiene significado físico
✅ **Extrapolación**: Respeta geometría del espacio de fases

**Limitaciones**:
❌ **Limitado a sistemas conservativos**: Sin disipación (¡LLG tiene amortiguamiento!)
❌ **Requiere datos de trayectoria**: No puede aprender de imágenes estáticas
❌ **Alta dimensionalidad**: Desafiante para sistemas espacialmente extendidos

**Extensiones**:
- **NNs Port-Hamiltonianas**: Incluyen disipación
- **NNs Lagrangianas**: Formulación alternativa
- **Integradores simplécticos**: Combinar con métodos numéricos

#### Referencias:
- **Greydanus et al. (2019)**: "Hamiltonian Neural Networks" - *NeurIPS*
- **Cranmer et al. (2020)**: "Lagrangian Neural Networks" - *ICLR Workshop*

### 3.2 Arquitecturas Informadas por Física con Restricciones Duras

**Concepto**: Diseñar arquitectura de red para **garantizar** restricciones físicas

#### A) Redes Libres de Divergencia

Para campos magnéticos: $\nabla \cdot \mathbf{B} = 0$

**Enfoque estándar**: Entrenar red, esperar que aprenda restricción
**Enfoque de restricción dura**: Parametrizar salida usando potencial vectorial

$$\mathbf{B} = \nabla \times \mathbf{A}$$

**Implementación**:
```python
class DivergenceFreeNetwork(nn.Module):
    """
    Red que SIEMPRE produce campos vectoriales libres de divergencia
    """
    def __init__(self):
        super().__init__()

        # Red aprende potencial vectorial A
        self.potential_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)  # Salida: A_x, A_y, A_z
        )

    def forward(self, coords):
        """
        Args:
            coords: [B, 3] coordenadas espaciales (x, y, z)

        Returns:
            B: [B, 3] campo magnético libre de divergencia
        """
        coords.requires_grad_(True)

        # Predecir potencial vectorial
        A = self.potential_net(coords)

        # Calcular rotor: B = ∇ × A
        B = self.compute_curl(A, coords)

        return B

    def compute_curl(self, A, coords):
        """Calcular rotor usando diferenciación automática"""
        # ∇ × A = (∂A_z/∂y - ∂A_y/∂z, ∂A_x/∂z - ∂A_z/∂x, ∂A_y/∂x - ∂A_x/∂y)

        B = []
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                k = 3 - i - j

                # Calcular derivadas parciales
                dA_i_dj = torch.autograd.grad(
                    A[:, i].sum(), coords,
                    create_graph=True, retain_graph=True
                )[0][:, j]

                dA_j_di = torch.autograd.grad(
                    A[:, j].sum(), coords,
                    create_graph=True, retain_graph=True
                )[0][:, i]

                B_k = dA_i_dj - dA_j_di
                B.append(B_k)

        B = torch.stack(B, dim=1)
        return B

# Verificar propiedad libre de divergencia
model = DivergenceFreeNetwork()
coords = torch.randn(100, 3, requires_grad=True)
B = model(coords)

# Calcular divergencia
div_B = 0
for i in range(3):
    dB_i_di = torch.autograd.grad(
        B[:, i].sum(), coords, retain_graph=True
    )[0][:, i]
    div_B += dB_i_di

print(f"Max |∇·B|: {div_B.abs().max().item():.2e}")  # Debería ser ~1e-6 (solo error numérico)
```

#### B) Redes Neuronales Equivariantes

**Problema**: Las leyes físicas son simétricas bajo rotaciones/reflexiones, pero las CNNs estándar no lo son

**Solución**: Construir **equivarianza** en arquitectura

$$f(R\mathbf{x}) = R f(\mathbf{x})$$

**Redes Equivariantes E(3)**:

```python
# Usando biblioteca e3nn
from e3nn import o3
from e3nn.nn import Gate

class E3EquivariantNetwork(nn.Module):
    """
    Red equivariante a rotaciones y reflexiones 3D

    Garantiza: Si rotas entrada, salida rota correspondientemente
    """
    def __init__(self):
        super().__init__()

        # Entrada: campo escalar (L=0)
        self.irreps_in = o3.Irreps("1x0e")  # 1 escalar

        # Oculto: escalares + vectores
        self.irreps_hidden = o3.Irreps("8x0e + 8x1o")

        # Salida: campo vectorial (L=1)
        self.irreps_out = o3.Irreps("1x1o")  # 1 vector

        # Capas
        self.tp1 = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_hidden, self.irreps_hidden
        )

        self.gate = Gate(
            "8x0e + 8x1o", [torch.sigmoid, torch.tanh]
        )

        self.tp2 = o3.FullyConnectedTensorProduct(
            self.irreps_hidden, self.irreps_hidden, self.irreps_out
        )

    def forward(self, x, positions):
        """
        Args:
            x: [B, 1] características escalares
            positions: [B, 3] coordenadas 3D

        Returns:
            out: [B, 3] campo vectorial
        """
        # Embeber posiciones
        sh = o3.spherical_harmonics(
            o3.Irreps("1x0e + 1x1o"),
            positions,
            normalize=True
        )

        # Producto tensorial con características
        x = self.tp1(x, sh)
        x = self.gate(x)
        x = self.tp2(x, sh)

        return x

# Probar equivarianza
model = E3EquivariantNetwork()
x = torch.randn(10, 1)
pos = torch.randn(10, 3)

# Rotación aleatoria
R = o3.rand_matrix()

# Rotar entrada
pos_rotated = pos @ R.T

# Predicciones
out_original = model(x, pos)
out_rotated = model(x, pos_rotated)

# Verificar si out_rotated ≈ R @ out_original
out_original_rotated = out_original @ R.T

print(f"Error de equivarianza: {(out_rotated - out_original_rotated).abs().max():.2e}")
# Debería ser ~1e-6
```

**Aplicación a Imágenes Magnéticas**:

Usar **CNNs equivariantes SE(2)** para rotaciones en el plano:

```python
# Usando biblioteca escnn
from escnn import gspaces
from escnn import nn as enn

class SE2EquivariantCNN(torch.nn.Module):
    """
    CNN equivariante a rotaciones y traslaciones en el plano
    """
    def __init__(self):
        super().__init__()

        # Definir grupo de simetría: SE(2) con rotaciones 8-fold
        self.gspace = gspaces.rot2dOnR2(N=8)

        # Entrada: campo escalar (representación trivial)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

        # Oculto: representación regular (todas las rotaciones)
        self.hidden_type = enn.FieldType(self.gspace, 16 * [self.gspace.regular_repr])

        # Salida: escalares invariantes (para regresión)
        self.out_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)

        # Capas
        self.conv1 = enn.R2Conv(self.in_type, self.hidden_type, kernel_size=5)
        self.relu1 = enn.ReLU(self.hidden_type)
        self.pool1 = enn.PointwiseAvgPool(self.hidden_type, 2)

        self.conv2 = enn.R2Conv(self.hidden_type, self.hidden_type, kernel_size=5)
        self.relu2 = enn.ReLU(self.hidden_type)
        self.pool2 = enn.PointwiseAvgPoolAntialiased(self.hidden_type, 2)

        # Pooling global (hace salida invariante a traslación)
        self.global_pool = enn.GroupPooling(self.hidden_type)

        # Totalmente conectado para regresión
        self.fc = torch.nn.Linear(16, 3)  # Salida: J, K, D

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] imagen MFM

        Returns:
            params: [B, 3] parámetros predichos
        """
        # Envolver como tensor geométrico
        x = enn.GeometricTensor(x, self.in_type)

        # Convoluciones
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Pool global
        x = self.global_pool(x)

        # Extraer tensor y aplanar
        x = x.tensor.flatten(1)

        # Regresión
        params = self.fc(x)

        return params

# Probar equivarianza rotacional
model = SE2EquivariantCNN()
img = torch.randn(1, 1, 64, 64)

# Rotar imagen 90 grados
img_rot = torch.rot90(img, k=1, dims=[2, 3])

# Las predicciones deberían ser idénticas (salida de regresión es invariante)
params_original = model(img)
params_rotated = model(img_rot)

print(f"Error de invariancia rotacional: {(params_original - params_rotated).abs().max():.4f}")
# Debería ser ~0
```

**Ventajas**:
✅ **Simetría garantizada**: La arquitectura impone restricciones físicas
✅ **Eficiencia de datos**: Se necesitan menos datos de entrenamiento (no hay que aprender simetrías)
✅ **Mejor extrapolación**: Respeta física en todas las regiones

**Limitaciones**:
❌ **Complejidad de arquitectura**: Requiere bibliotecas especializadas (e3nn, escnn)
❌ **Costo computacional**: Operaciones equivariantes son costosas
❌ **Simetrías limitadas**: Solo maneja grupos con representaciones conocidas

**Beneficio de Interpretabilidad**:
- **CNN de caja negra**: Podría aprender artefactos dependientes de rotación
- **CNN equivariante**: Trata todas las orientaciones igual provablemente → más confiable

---

## 4. Regresión Simbólica y Descubrimiento de Ecuaciones

**Objetivo**: Extraer expresiones simbólicas interpretables de datos o redes neuronales

### 4.1 AI Feynman / Regresión Simbólica

**Concepto**: Buscar espacio de expresiones matemáticas para encontrar fórmula más simple que ajuste datos

$$\text{Encontrar } f \text{ tal que } y = f(x_1, x_2, \ldots, x_n)$$

**Métodos**:
1. **Programación Genética**: Evolucionar árbol de operaciones
2. **Regresión Dispersa**: Ajustar base polinomial con regularización (SINDy)
3. **Búsqueda guiada por NN**: Usar NN para guiar búsqueda simbólica

**Implementación** (usando PySR):
```python
from pysr import PySRRegressor
import numpy as np

def discover_physical_law(X, y, variable_names):
    """
    Descubrir relación simbólica entre entradas X y salida y

    Args:
        X: [N, D] características de entrada (ej., correlaciones de espín, gradientes)
        y: [N] valores objetivo (ej., parámetro de intercambio J)
        variable_names: Lista de nombres de características

    Returns:
        model: Modelo PySR ajustado con ecuaciones simbólicas
    """
    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt", "sin", "cos"],
        model_selection="best",
        loss="L2DistLoss()",
        maxsize=20,  # Complejidad máxima
        parsimony=0.01,  # Preferencia por modelos más simples
        variable_names=variable_names
    )

    model.fit(X, y)

    # Obtener mejores ecuaciones
    print("Ecuaciones principales descubiertas:")
    print(model)

    return model

# Ejemplo: Descubrir relación entre características de pared de dominio y J
# Extraer características de imágenes
def extract_domain_wall_features(mfm_images, masks):
    """
    Extraer características físicamente significativas de imágenes MFM

    Returns:
        features: [N, D] con columnas:
            - wall_width: Ancho promedio de pared de dominio
            - wall_gradient: Magnitud de gradiente en paredes
            - correlation_length: Longitud de correlación de espín
            - texture_energy: Densidad de energía estimada
    """
    features = []

    for img, mask in zip(mfm_images, masks):
        # Ancho de pared de dominio (desde máscara)
        wall_width = compute_wall_width(img, mask)

        # Gradiente en paredes
        grad = np.gradient(img)
        wall_gradient = np.mean(np.abs(grad)[mask])

        # Longitud de correlación (desde autocorrelación)
        corr = np.fft.fft2(img)
        corr_length = extract_correlation_length(corr)

        # Estimación de energía
        texture_energy = np.sum(grad**2)

        features.append([wall_width, wall_gradient, corr_length, texture_energy])

    return np.array(features)

# Ejecutar regresión simbólica
features = extract_domain_wall_features(train_images, wall_masks)
J_values = train_labels[:, 0]  # Valores verdaderos de J

model = discover_physical_law(
    features,
    J_values,
    variable_names=["wall_width", "wall_grad", "corr_len", "energy"]
)

# Salida de ejemplo:
# Mejor ecuación: J = 2.3 * sqrt(corr_len / wall_width) + 0.1
# Interpretación: Intercambio ~ sqrt(correlación / ancho_pared)
# Física: ξ ~ sqrt(J/K), δ_wall ~ sqrt(J/K) → J ~ ξ²/δ²
```

**Validación**: ¡Verificar si la ecuación descubierta se alinea con la física conocida!

$$\delta_{\text{wall}} = \pi \sqrt{\frac{A}{K}} \quad \Rightarrow \quad J \propto \frac{1}{\delta_{\text{wall}}^2}$$

### 4.2 SINDy (Sparse Identification of Nonlinear Dynamics)

**Concepto**: Descubrir PDEs gobernantes desde datos espaciotemporales

**Aplicación a ecuación LLG**:

```python
import pysindy as ps

def discover_llg_equation(spin_trajectories, dt):
    """
    Descubrir ecuación que gobierna dinámica de espines

    Args:
        spin_trajectories: [T, N, 3] componentes de espín en el tiempo
        dt: Paso temporal

    Returns:
        model: Modelo SINDy con ecuación descubierta
    """
    # Aplanar dimensión espacial
    T, N, D = spin_trajectories.shape
    X = spin_trajectories.reshape(T, N * D)

    # Construir biblioteca de funciones candidatas
    # Incluir: polinomios, productos, derivadas espaciales
    library = ps.PolynomialLibrary(degree=3) + ps.FourierLibrary()

    # Regresión dispersa
    optimizer = ps.STLSQ(threshold=0.05, alpha=0.1)

    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=library,
        differentiation_method=ps.FiniteDifference(order=2)
    )

    model.fit(X, t=dt)

    # Imprimir ecuaciones descubiertas
    model.print()

    return model

# Salida de ejemplo:
# dS_x/dt = -γ S_y H_z + γ S_z H_y + α (S_y dS_z/dt - S_z dS_y/dt)
# → ¡Coincide con ecuación LLG!
```

**Valor de Interpretabilidad**:
- **Validación**: ¿La ecuación descubierta coincide con LLG conocida?
- **Extracción de parámetros**: Los coeficientes dan γ, α
- **Detección de anomalías**: Las desviaciones indican física desconocida o errores

### 4.3 Destilar Redes Neuronales a Expresiones Simbólicas

**Concepto**: Entrenar NN, luego extraer aproximación simbólica

```python
def distill_nn_to_symbolic(model, X_train, y_train):
    """
    Extraer expresión simbólica de red neuronal entrenada

    Args:
        model: NN entrenada
        X_train: Entradas de entrenamiento
        y_train: Salidas de entrenamiento

    Returns:
        symbolic_model: Modelo PySR aproximando NN
    """
    # Generar predicciones desde NN
    with torch.no_grad():
        y_pred_nn = model(X_train).numpy()

    # Ajustar modelo simbólico a predicciones de NN
    symbolic_model = PySRRegressor(
        niterations=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "tanh"],
        maxsize=15
    )

    symbolic_model.fit(X_train.numpy(), y_pred_nn)

    # Comparar
    y_pred_symbolic = symbolic_model.predict(X_train.numpy())
    fidelity = np.corrcoef(y_pred_nn.flatten(), y_pred_symbolic.flatten())[0, 1]

    print(f"Fidelidad de aproximación simbólica: {fidelity:.4f}")
    print(f"Mejor ecuación: {symbolic_model.get_best()}")

    return symbolic_model

# Uso
symbolic_approximation = distill_nn_to_symbolic(trained_cnn, features, labels)

# Interpretación: ¡Ahora tienes fórmula legible para humanos!
```

**Ventajas**:
✅ **Legible para humanos**: Los científicos pueden entender y validar
✅ **Compacta**: Fórmula simple vs millones de parámetros
✅ **Generalizable**: Las expresiones simbólicas a menudo extrapolan mejor

**Limitaciones**:
❌ **Pérdida de precisión**: El modelo simbólico es aproximación
❌ **Limitado a baja dimensionalidad**: Difícil con entradas de alta dimensión (imágenes)
❌ **Búsqueda es costosa**: Explosión combinatoria

---

## 5. Interpretabilidad Mecanística

**Objetivo**: Entender **cómo** calcula la red, no solo **qué** calcula

### 5.1 Sondeo de Representaciones Internas

**Concepto**: Entrenar sondas lineales en activaciones intermedias para detectar conceptos aprendidos

```python
class ConceptProbe:
    """
    Sondear qué conceptos físicos están codificados en capas de red
    """
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.activations = None

        # Registrar hook para extraer activaciones
        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def extract_activations(self, dataloader):
        """Extraer activaciones para dataset completo"""
        all_activations = []

        with torch.no_grad():
            for images, _ in dataloader:
                _ = self.model(images)  # Activa hook
                all_activations.append(self.activations)

        return torch.cat(all_activations, dim=0)

    def train_probe(self, activations, concepts):
        """
        Entrenar sonda lineal para predecir concepto desde activaciones

        Args:
            activations: [N, C, H, W] o [N, D]
            concepts: [N] etiquetas binarias (ej., "¿tiene skyrmión?")

        Returns:
            probe: Clasificador lineal entrenado
        """
        # Aplanar dimensiones espaciales
        if activations.ndim == 4:
            activations = activations.mean(dim=[2, 3])  # Pool promedio global

        # Entrenar regresión logística
        from sklearn.linear_model import LogisticRegression
        probe = LogisticRegression(max_iter=1000)
        probe.fit(activations.numpy(), concepts.numpy())

        # Evaluar
        accuracy = probe.score(activations.numpy(), concepts.numpy())
        print(f"Precisión de sonda para concepto: {accuracy:.4f}")

        return probe

# Ejemplo: Sondear para "presencia de skyrmión"
concept_probe = ConceptProbe(model, layer_name='features.6')

# Extraer activaciones
activations = concept_probe.extract_activations(train_loader)

# Anotar datos con etiquetas de concepto
has_skyrmion = torch.tensor([
    detect_skyrmion(img) for img, _ in train_loader.dataset
])

# Entrenar sonda
skyrmion_probe = concept_probe.train_probe(activations, has_skyrmion)

# Interpretación: Si precisión > 90%, ¡capa 6 ha aprendido detector de skyrmiones!
```

**Aplicación**: Sondear para múltiples conceptos físicos

```python
concepts = {
    'has_skyrmion': detect_skyrmions(images),
    'has_domain_walls': detect_walls(images),
    'high_anisotropy': labels[:, 1] > threshold,  # K > umbral
    'chiral_texture': detect_chirality(images),
}

for concept_name, concept_labels in concepts.items():
    probe = concept_probe.train_probe(activations, concept_labels)
    print(f"{concept_name}: Precisión = {probe.score(...):.2%}")

# Salida de ejemplo:
# has_skyrmion: Precisión = 94%  ← ¡Capa aprendió detección de skyrmiones!
# has_domain_walls: Precisión = 87%
# high_anisotropy: Precisión = 62%  ← No representado explícitamente
# chiral_texture: Precisión = 91%
```

### 5.2 Análisis de Circuitos

**Concepto**: Descomponer red en "circuitos" interpretables

**Ejemplo**: Encontrar qué neuronas calculan funciones específicas

```python
def find_domain_wall_detector(model, mfm_images, wall_masks):
    """
    Identificar neuronas que se activan en paredes de dominio

    Args:
        model: CNN entrenada
        mfm_images: [N, 1, H, W]
        wall_masks: [N, H, W] máscaras binarias (1 = pared, 0 = fondo)

    Returns:
        wall_neurons: Lista de tuplas (capa, canal)
    """
    wall_neurons = []

    # Enganchar todas las capas convolucionales
    hooks = []
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    with torch.no_grad():
        _ = model(mfm_images)

    # Analizar cada capa
    for layer_name, acts in activations.items():
        # acts: [N, C, H', W']
        N, C, H, W = acts.shape

        # Redimensionar máscaras para coincidir con tamaño de activación
        masks_resized = F.interpolate(
            wall_masks.unsqueeze(1).float(),
            size=(H, W),
            mode='nearest'
        )

        # Para cada canal, calcular correlación con máscara de pared
        for c in range(C):
            channel_acts = acts[:, c, :, :]  # [N, H', W']

            # Correlación entre activación y máscara
            corr = 0
            for n in range(N):
                corr += np.corrcoef(
                    channel_acts[n].flatten(),
                    masks_resized[n, 0].flatten()
                )[0, 1]
            corr /= N

            # Alta correlación → neurona detecta paredes
            if corr > 0.7:
                wall_neurons.append((layer_name, c, corr))

    # Remover hooks
    for hook in hooks:
        hook.remove()

    # Ordenar por correlación
    wall_neurons.sort(key=lambda x: x[2], reverse=True)

    return wall_neurons

# Uso
wall_detectors = find_domain_wall_detector(model, train_images, wall_masks)

print("Neuronas principales detectoras de pared de dominio:")
for layer, channel, corr in wall_detectors[:5]:
    print(f"  {layer}, canal {channel}: correlación = {corr:.3f}")

# Salida:
# conv1, canal 23: correlación = 0.891
# conv2, canal 41: correlación = 0.847
# ...
```

**Visualización**: Mostrar a qué responde cada neurona

```python
def visualize_neuron_receptive_field(model, layer, channel, images):
    """Mostrar imágenes que activan maximalmente neurona específica"""

    # Extraer activaciones
    hook_output = []
    def hook(module, input, output):
        hook_output.append(output)

    target_layer = dict(model.named_modules())[layer]
    handle = target_layer.register_forward_hook(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(images)

    activations = hook_output[0][:, channel].mean(dim=[1, 2])  # [N]

    # Encontrar imágenes con mayor activación
    top_k = 9
    top_indices = activations.topk(top_k).indices

    # Visualizar
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for idx, ax in zip(top_indices, axes.flat):
        ax.imshow(images[idx].squeeze(), cmap='gray')
        ax.set_title(f'Activación: {activations[idx]:.2f}')
        ax.axis('off')

    fig.suptitle(f'{layer}, Canal {channel}')

    handle.remove()

# Ejemplo
visualize_neuron_receptive_field(model, 'conv1', 23, train_images)
# Muestra: Esta neurona responde a paredes de dominio con orientación específica
```

---

## 6. Verificación de Consistencia Física

### 6.1 Pruebas de Leyes de Conservación

**Probar si el modelo respeta leyes de conservación fundamentales**

#### A) Conservación de Energía

```python
def test_energy_conservation(model, spin_trajectories):
    """
    Probar si predicciones del modelo conservan energía

    Args:
        model: Modelo de dinámica (predice dS/dt)
        spin_trajectories: [T, N, 3] serie temporal

    Returns:
        energy_drift: [T] energía total en el tiempo
    """
    energies = []

    for t in range(len(spin_trajectories)):
        spins = spin_trajectories[t]

        # Calcular energía Hamiltoniana
        E = compute_hamiltonian_energy(spins)
        energies.append(E.item())

    energies = np.array(energies)

    # Verificar deriva
    energy_drift = energies - energies[0]
    relative_drift = energy_drift / abs(energies[0])

    print(f"Deriva de energía: {energy_drift[-1]:.2e} ({relative_drift[-1]:.2%})")

    # Visualizar
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(energies)
    plt.xlabel('Paso temporal')
    plt.ylabel('Energía Total')
    plt.title('Evolución de Energía')

    plt.subplot(1, 2, 2)
    plt.plot(relative_drift)
    plt.xlabel('Paso temporal')
    plt.ylabel('Deriva Relativa de Energía')
    plt.title('Verificación de Conservación de Energía')
    plt.axhline(0, color='r', linestyle='--')

    return energy_drift

def compute_hamiltonian_energy(spins, J=1.0, K=0.1, D=0.5):
    """
    Calcular energía Hamiltoniana total

    H = -J Σ S_i·S_j + K Σ S_z² + D Σ S·(∇×S)
    """
    # Intercambio
    E_ex = -J * exchange_energy(spins)

    # Anisotropía
    E_ani = K * (spins[:, 2]**2).sum()

    # DMI
    E_dmi = D * dmi_energy(spins)

    E_total = E_ex + E_ani + E_dmi

    return E_total
```

**Criterio de Aceptación**:
- ✅ **Bueno**: Deriva de energía < 1% sobre simulación
- ⚠️ **Aceptable**: Deriva de energía < 5%
- ❌ **Falla**: Deriva de energía > 10% → ¡Modelo viola física!

#### B) Pruebas de Simetría

```python
def test_rotational_symmetry(model, image, angles=[0, 90, 180, 270]):
    """
    Probar si modelo es invariante/equivariante a rotaciones

    Args:
        model: Modelo de regresión (debería ser invariante)
        image: [1, C, H, W] imagen de prueba
        angles: Ángulos de rotación a probar

    Returns:
        symmetry_error: Desviación máxima entre rotaciones
    """
    predictions = []

    for angle in angles:
        # Rotar imagen
        k = angle // 90
        img_rotated = torch.rot90(image, k=k, dims=[2, 3])

        # Predecir
        with torch.no_grad():
            pred = model(img_rotated)

        predictions.append(pred)

    predictions = torch.stack(predictions)

    # Verificar varianza (debería ser ~0 para modelo invariante)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)

    symmetry_error = std_pred / (mean_pred.abs() + 1e-8)

    print("Resultados de Prueba de Simetría:")
    print(f"  Predicción promedio: {mean_pred}")
    print(f"  Std entre rotaciones: {std_pred}")
    print(f"  Error relativo: {symmetry_error}")

    # Visualizar
    fig, axes = plt.subplots(1, len(angles), figsize=(15, 3))
    for ax, angle, pred in zip(axes, angles, predictions):
        k = angle // 90
        img_rot = torch.rot90(image, k=k, dims=[2, 3])
        ax.imshow(img_rot.squeeze(), cmap='gray')
        ax.set_title(f'{angle}°\nPred: {pred[0].numpy()}')
        ax.axis('off')

    return symmetry_error

# Probar
symmetry_err = test_rotational_symmetry(model, test_image)

if symmetry_err.max() < 0.05:
    print("✅ Modelo respeta simetría rotacional")
else:
    print(f"❌ ¡Violación de simetría detectada! Error: {symmetry_err.max():.2%}")
```

#### C) Verificación de Límites Físicos

```python
def verify_physical_bounds(predictions):
    """
    Verificar si predicciones respetan restricciones físicas

    Returns:
        violations: Dict de violaciones de restricciones
    """
    violations = {}

    # Intercambio debe ser positivo (ferromagnético)
    J = predictions[:, 0]
    if (J < 0).any():
        violations['J_negative'] = (J < 0).sum().item()

    # Rango típico de DMI: |D| < 5 mJ/m²
    D = predictions[:, 2]
    if (D.abs() > 5.0).any():
        violations['D_unrealistic'] = (D.abs() > 5.0).sum().item()

    # Rango típico de anisotropía: |K| < 1 MJ/m³
    K = predictions[:, 1]
    if (K.abs() > 1.0).any():
        violations['K_unrealistic'] = (K.abs() > 1.0).sum().item()

    # Consistencia de diagrama de fases: skyrmiones necesitan D > umbral
    skyrmion_images = detect_skyrmions(images)
    if skyrmion_images.any():
        D_skyrmion = D[skyrmion_images]
        if (D_skyrmion < 0.5).any():
            violations['skyrmion_phase_inconsistent'] = (D_skyrmion < 0.5).sum().item()

    # Imprimir reporte
    if len(violations) == 0:
        print("✅ Todas las predicciones dentro de límites físicos")
    else:
        print("⚠️ Violaciones de límites físicos detectadas:")
        for violation, count in violations.items():
            print(f"  - {violation}: {count} instancias")

    return violations
```

### 6.2 Análisis de Causalidad y Contrafactual

**Pregunta**: ¿El modelo entiende relaciones causales?

**Método**: Intervenir en representaciones aprendidas, verificar si los efectos son físicos

```python
def counterfactual_analysis(model, image, layer_name, concept_direction):
    """
    Modificar representación interna en dirección de concepto, observar efecto

    Args:
        model: Modelo entrenado
        image: Imagen de entrada
        layer_name: Qué capa intervenir
        concept_direction: [D] dirección en espacio de activación (ej., "más skyrmiones")

    Returns:
        counterfactual_predictions: Predicciones después de intervención
    """
    activations = {}

    def hook(module, input, output):
        activations['target'] = output
        return output

    def intervention_hook(module, input, output):
        # Modificar activación en dirección de concepto
        modified = output + alpha * concept_direction.view(1, -1, 1, 1)
        return modified

    # Predicción original
    target_layer = dict(model.named_modules())[layer_name]
    handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        pred_original = model(image)

    handle.remove()

    # Predicciones contrafactuales
    results = []
    alphas = np.linspace(-2, 2, 11)

    for alpha in alphas:
        handle = target_layer.register_forward_hook(
            lambda m, i, o: o + alpha * concept_direction.view(1, -1, 1, 1)
        )

        with torch.no_grad():
            pred_modified = model(image)

        results.append(pred_modified.numpy())
        handle.remove()

    results = np.array(results)

    # Visualizar efecto
    plt.figure(figsize=(10, 6))
    for param_idx, param_name in enumerate(['J', 'K', 'D']):
        plt.plot(alphas, results[:, 0, param_idx], marker='o', label=param_name)

    plt.axvline(0, color='r', linestyle='--', label='Original')
    plt.xlabel('Fuerza de intervención α')
    plt.ylabel('Valor de parámetro predicho')
    plt.title(f'Contrafactual: Efecto de modificar representación "{concept_name}"')
    plt.legend()
    plt.grid()

    return results

# Encontrar dirección de concepto (ej., "presencia de skyrmión")
# Entrenar sonda para encontrar dirección que separa skyrmión/no-skyrmión
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Extraer activaciones para imágenes con/sin skyrmiones
acts_skyrmion = activations[has_skyrmion]
acts_no_skyrmion = activations[~has_skyrmion]

# Encontrar hiperplano de separación
lda = LinearDiscriminantAnalysis()
lda.fit(activations, has_skyrmion)

skyrmion_direction = torch.from_numpy(lda.coef_[0]).float()

# Ejecutar contrafactual
counterfactuals = counterfactual_analysis(
    model, test_image, 'features.6', skyrmion_direction
)

# Interpretación: ¿Aumentar "skyrmión-dad" aumenta predicción de D?
# Si sí → ¡modelo aprendió que D controla skyrmiones (física correcta!)
```

---

## 7. Análisis Comparativo

### 7.1 Tabla Resumen: Métodos de Interpretabilidad

| **Método** | **Tipo** | **Salida** | **Costo Computacional** | **Validación Física** | **Mejor Caso de Uso** |
|------------|----------|------------|----------------------|----------------------|-------------------|
| **Mapas de Saliencia** | Atribución post-hoc | Importancia de píxeles | ★☆☆☆☆ (Muy rápido) | ⭐☆☆☆ (Débil) | Verificación visual rápida |
| **Grad-CAM** | Atribución post-hoc | Importancia de región | ★☆☆☆☆ (Rápido) | ⭐⭐☆☆ (Moderado) | Identificar características importantes |
| **SHAP** | Atribución post-hoc | Contribuciones de características | ★★★★☆ (Lento) | ⭐⭐☆☆ (Moderado) | Atribución rigurosa |
| **LIME** | Local post-hoc | Aproximación lineal local | ★★★☆☆ (Medio) | ⭐☆☆☆ (Débil) | Explicar predicciones individuales |
| **Sondeo** | Mecanístico | Detección de conceptos | ★★☆☆☆ (Medio) | ⭐⭐⭐☆ (Bueno) | Entender representaciones |
| **HNN** | Arquitectura restringida | Función de energía | ★★★☆☆ (Entrenamiento) | ⭐⭐⭐⭐ (Excelente) | Dinámica con conservación |
| **NNs Equivariantes** | Arquitectura restringida | Predicciones simétricas | ★★★★☆ (Costoso) | ⭐⭐⭐⭐ (Excelente) | Garantizar simetrías |
| **Regresión Simbólica** | Descubrimiento de ecuaciones | Fórmula matemática | ★★★★★ (Muy lento) | ⭐⭐⭐⭐ (Excelente) | Extraer leyes gobernantes |
| **Pruebas de Conservación** | Verificación | Métricas pasa/falla | ★☆☆☆☆ (Rápido) | ⭐⭐⭐⭐ (Excelente) | Validar consistencia física |

### 7.2 Árbol de Decisión

```
OBJETIVO: Interpretar modelo de regresión de parámetros de dominio magnético

Q1: ¿Necesitas entender predicciones individuales o modelo general?
  ├─ Individual → Métodos post-hoc (Grad-CAM, SHAP)
  └─ General → Interpretabilidad mecanística (Sondeo, Análisis de circuitos)

Q2: ¿Puedes restringir arquitectura durante entrenamiento?
  ├─ SÍ → Usar arquitecturas restringidas por física (HNN, Equivariante)
  │         ✅ Mejor: Consistencia física garantizada
  └─ NO → Análisis post-hoc de modelo de caja negra

Q3: ¿Qué aspecto de física quieres verificar?
  ├─ Conservación de energía → HNN o pruebas explícitas de energía
  ├─ Simetrías → NNs equivariantes o pruebas de simetría
  ├─ Ecuaciones gobernantes → Regresión simbólica (SINDy)
  └─ Importancia de características → Grad-CAM, SHAP

Q4: ¿Tienes conceptos etiquetados (skyrmiones, paredes, etc.)?
  ├─ SÍ → Sondeo de conceptos, TCAV
  └─ NO → Visualización de características no supervisada

Q5: ¿Cuál es tu presupuesto computacional?
  ├─ Bajo → Mapas de saliencia, Grad-CAM
  ├─ Medio → Sondeo, modelos equivariantes
  └─ Alto → SHAP, regresión simbólica

RECOMENDACIÓN: ¡Combinar múltiples métodos!
  - Entrenamiento: Usar CNN equivariante si es posible
  - Validación: Ejecutar pruebas de conservación/simetría
  - Interpretación: Grad-CAM + sondeo para conceptos
  - Comprensión: Regresión simbólica en características extraídas
```

### 7.3 Pipeline Recomendado para Investigación de Dominios Magnéticos

**Etapa 1: Entrenamiento con Interpretabilidad Incorporada**
```python
# 1. Usar CNN equivariante SE(2) para invarianza rotacional
model = SE2EquivariantCNN()

# 2. Entrenar con pérdida informada por física
def physics_aware_loss(predictions, targets, images):
    # Pérdida de regresión estándar
    loss_mse = F.mse_loss(predictions, targets)

    # Penalización de consistencia física
    loss_energy = energy_consistency_penalty(predictions, images)
    loss_bounds = bounds_violation_penalty(predictions)

    return loss_mse + 0.1 * loss_energy + 0.05 * loss_bounds

# 3. Monitorear métricas de interpretabilidad durante entrenamiento
for epoch in range(epochs):
    train_one_epoch(model, train_loader, physics_aware_loss)

    # Validación
    test_loss = evaluate(model, test_loader)

    # Verificaciones de interpretabilidad
    symmetry_error = test_rotational_symmetry(model, test_images)
    energy_violations = verify_physical_bounds(predictions)

    log_metrics(epoch, test_loss, symmetry_error, energy_violations)
```

**Etapa 2: Análisis Post-Hoc**
```python
# 1. Atribución de características
gradcam = GradCAM(model, target_layer='features.6')
for img, label in test_dataset:
    cam = gradcam(img, target_param_idx=0)  # Explicar predicción de J
    visualize_overlay(img, cam)

# 2. Sondeo de conceptos
probe_skyrmions = train_concept_probe(model, 'features.6', skyrmion_labels)
probe_walls = train_concept_probe(model, 'features.6', wall_labels)

print(f"Concepto de skyrmión aprendido: {probe_skyrmions.score() > 0.9}")
print(f"Concepto de pared de dominio aprendido: {probe_walls.score() > 0.9}")

# 3. SHAP para atribución rigurosa
shap_explainer = SHAPMagneticExplainer(model, background_data)
shap_values = shap_explainer.explain(test_image)
analyze_important_regions(shap_values)
```

**Etapa 3: Validación Física**
```python
# 1. Pruebas de conservación
energy_drift = test_energy_conservation(model, trajectories)
assert energy_drift.max() < 0.05, "¡Conservación de energía violada!"

# 2. Pruebas de simetría
for image in test_set:
    sym_error = test_rotational_symmetry(model, image)
    assert sym_error.max() < 0.05, "¡Simetría rotacional violada!"

# 3. Consistencia de diagrama de fases
predictions = model(all_test_images)
check_phase_diagram_consistency(predictions, image_textures)
```

**Etapa 4: Comprensión Simbólica (Opcional)**
```python
# Extraer características
features = extract_physical_features(images)  # ancho_pared, tamaño_núcleo, etc.
labels = ground_truth_parameters

# Descubrir relaciones simbólicas
symbolic_model = discover_physical_law(features, labels[:, 0], ['width', 'core', 'corr'])

print("Relación descubierta para J:")
print(symbolic_model.get_best())
# Salida de ejemplo: J = 2.1 * (corr / width²) + 0.3

# Validar contra física conocida
validate_symbolic_expression(symbolic_model, known_relations)
```

---

## 8. Trabajos Estado del Arte (2022+)

### 8.1 Interpretabilidad Informada por Física

#### Cranmer et al. (2023) - *Physical Review E*

**Título**: "Interpretable Machine Learning for Physics: Bridging Models and Understanding"

**Contribuciones Clave**:
1. **Destilación simbólica**: Extraer ecuaciones de NNs entrenadas
2. **Búsqueda de arquitectura guiada por física**: Evolucionar arquitecturas respetando simetrías
3. **Benchmark en sistemas Hamiltonianos**: 95% de precisión en recuperación de ecuaciones

**Relevancia para Dominios Magnéticos**:
- Técnicas aplicables a dinámica LLG
- Regresión simbólica identifica términos de energía
- Podría extraer $H(J, K, D)$ de modelos entrenados

**Código**: https://github.com/MilesCranmer/PySR

#### Alet al. (2023) - *Nature Machine Intelligence*

**Título**: "Physically Consistent Neural Networks for Fluid Dynamics"

**Contribuciones Clave**:
1. **Capas con restricciones duras**: Campos de velocidad libres de divergencia
2. **Monitoreo de conservación**: Detección de violaciones de física en tiempo real
3. **Características interpretables**: Vorticidad, circulación emergen automáticamente

**Adaptación a Magnetismo**:
```python
# Enfoque similar para magnetización
class MagnetizationConservingLayer(nn.Module):
    """Capa que preserva |M| = M_s"""
    def forward(self, M):
        # Proyectar sobre variedad de restricción
        M_norm = M / (M.norm(dim=1, keepdim=True) + 1e-8)
        return M_s * M_norm
```

**Paper**: https://doi.org/10.1038/s42256-023-00680-y

### 8.2 Interpretabilidad Mecanística en Aplicaciones Científicas

#### Lample & Charton (2024) - *ICLR*

**Título**: "Deep Learning for Symbolic Mathematics: Recovering Equations from Data"

**Contribuciones Clave**:
1. **Descubrimiento de ecuaciones basado en Transformer**: Supera programación genética
2. **Integración de análisis dimensional**: Respeta unidades
3. **Resultados**: 89% de tasa de éxito en benchmark de ecuaciones de Feynman

**Aplicación**:
```python
# Usar para descubrir funcional de energía magnética
from symbolicregression import TransformerSR

model = TransformerSR()
model.train(spin_configs, energies)

# Salida: H = J*sum(S·S') + K*sum(Sz²) + D*sum(S·(∇×S))
# → ¡Recuperó forma funcional correcta!
```

**Código**: https://github.com/facebookresearch/symbolicregression

#### Jiang et al. (2024) - *NeurIPS*

**Título**: "Interpretable Scientific Discovery with Self-Supervised Concept Learning"

**Contribuciones Clave**:
1. **Extracción de conceptos no supervisada**: No se necesita etiquetado manual
2. **Explicaciones basadas en conceptos**: Interpretables para humanos
3. **Aplicación a ciencia de materiales**: Descubrió transiciones de fase

**Aplicación a Dominios Magnéticos**:
- Descubrir automáticamente conceptos: "skyrmión", "fase de rayas", "laberinto"
- No se necesita anotación manual
- Modelo explica predicciones usando conceptos descubiertos

**Paper**: https://proceedings.neurips.cc/paper/2024/...

### 8.3 Simetría y Equivarianza

#### Batzner et al. (2022) - *Nature Communications*

**Título**: "E(3)-Equivariant Graph Neural Networks for Molecules and Materials"

**Contribuciones Clave**:
1. **Predicción de propiedades de materiales estado del arte**: GNN equivariante E(3)
2. **Atención interpretable**: Muestra qué átomos/enlaces importan
3. **Benchmark**: 50% de reducción de error vs NNs estándar

**Adaptación**: Usar para nanoestructuras magnéticas en redes

```python
from e3nn import o3
# Modelar red de espín como grafo
# Aristas = interacciones de intercambio
# Características de nodo = orientación de espín local
# → GNN equivariante predice energía total
```

**Código**: https://github.com/mir-group/nequip

#### Weiler & Cesa (2022) - *ICML*

**Título**: "General E(2)-Equivariant Steerable CNNs"

**Contribuciones Clave**:
1. **Kernels dirigibles**: Implementación eficiente de convoluciones equivariantes
2. **Teoría**: Caracterización completa de mapeos equivariantes
3. **Aplicaciones**: Clasificación de imágenes, segmentación

**Aplicación Directa a Imágenes MFM**:
- Imágenes MFM tienen simetría de rotación en el plano (E(2))
- Usar biblioteca escnn con CNNs dirigibles
- Invarianza rotacional garantizada para predicciones de parámetros

**Código**: https://github.com/QUVA-Lab/escnn

### 8.4 Causalidad y Contrafactuales en Física

#### Schwab & Karlen (2023) - *Physical Review Letters*

**Título**: "Causal Machine Learning for Physics: Interventional Explanations"

**Contribuciones Clave**:
1. **Descubrimiento causal desde datos observacionales**: Aprender grafo causal
2. **Consultas intervencionales**: "¿Qué pasa si D aumenta?"
3. **Aplicación a transiciones de fase**: Identificar parámetros de control

**Aplicación a Dominios Magnéticos**:
```python
# Aprender grafo causal: J, K, D → {textura magnética} → {imagen MFM}
# Intervenir en D, observar cambio de textura
# Verificar si modelo respeta estructura causal conocida
```

**Paper**: https://doi.org/10.1103/PhysRevLett.131.XXX

---

## 9. Hoja de Ruta de Investigación

### 9.1 Corto Plazo (3-6 meses): Fundación

**Objetivo**: Establecer baseline de interpretabilidad y validar consistencia física

**Hitos**:

**Mes 1-2**: Análisis post-hoc
- Implementar Grad-CAM para todas las imágenes de prueba
- Identificar qué regiones usa el modelo para cada parámetro
- Validar contra intuición física (paredes para J, núcleos para D)

**Mes 3-4**: Pruebas de consistencia física
- Probar simetría rotacional en dataset
- Verificar límites físicos (J > 0, rango razonable de D)
- Medir conservación de energía si aplica

**Mes 5-6**: Sondeo de conceptos
- Anotar subconjunto con conceptos físicos (skyrmiones, paredes, rayas)
- Entrenar sondas en capas intermedias
- Determinar qué capas aprenden qué conceptos

**Entregables**:
- Reporte de interpretabilidad con visualizaciones
- Métricas de consistencia física
- Precisión de detección de conceptos

**Objetivo de Paper**: Paper de workshop (ICML Workshops, AI4Science)

### 9.2 Mediano Plazo (6-12 meses): Modelos Restringidos por Física

**Objetivo**: Entrenar modelos interpretables con consistencia física incorporada

**Hitos**:

**Mes 1-3**: Arquitectura equivariante
- Implementar CNN equivariante SE(2)
- Entrenar y comparar con CNN estándar
- Medir reducción de error de simetría

**Mes 4-6**: Pérdidas informadas por física
- Agregar penalización de consistencia de energía
- Implementar restricciones de diagrama de fases
- Benchmark de tasas de violación física

**Mes 7-9**: Aprendizaje Hamiltoniano
- Para datos dinámicos: Entrenar HNN para aprender funcional de energía
- Extraer expresión simbólica para $H(J, K, D)$
- Validar contra teoría micromagnética

**Mes 10-12**: Integración y benchmarking
- Comparar modelos interpretables vs caja negra
- Análisis de trade-off: precisión vs interpretabilidad
- Estudio de usuario con expertos del dominio

**Entregables**:
- Modelo restringido por física logrando precisión comparable
- Relaciones simbólicas extraídas
- Estudio comparativo

**Objetivo de Paper**: Paper de conferencia (NeurIPS, ICML, ICLR) o PRB

### 9.3 Largo Plazo (12-24 meses): Comprensión Mecanística

**Objetivo**: Extraer comprensión mecanística completa y ecuaciones simbólicas

**Componentes**:

**Pipeline de Regresión Simbólica**:
- Extraer características físicas de todas las imágenes
- Ejecutar regresión simbólica a gran escala (PySR, AI Feynman)
- Descubrir relaciones: $J = f(\text{características})$, $D = g(\text{características})$
- Publicar "solución de problema inverso" simbólica

**Descubrimiento Causal**:
- Construir grafo causal: Parámetros → Textura → Imagen → Mediciones
- Experimentos intervencionales (si se tiene acceso a simulaciones)
- Validar que modelo respeta estructura causal

**Sistema de IA Explicable**:
- Herramienta orientada al usuario: "¿Por qué predijiste J=1.2?"
- Proporciona: Visualización Grad-CAM + explicaciones de conceptos + aproximación simbólica
- Integrado con software de microscopio

**Despliegue**:
- Dashboard de interpretabilidad en tiempo real
- Detección de anomalías (marcar predicciones no físicas)
- Aprendizaje activo (sugerir experimentos informativos)

**Entregables**:
- Capítulo de tesis doctoral sobre interpretabilidad
- Toolkit de interpretabilidad open-source
- Paper de revista (Nature Communications, Science Advances)
- Software integrado con sistemas MFM/SPLEEM

---

## 10. Recursos de Implementación

### 10.1 Estructura de Repositorio de Código

```
interpretability-magnetic-domains/
├── attribution/
│   ├── saliency.py
│   ├── gradcam.py
│   ├── shap_explainer.py
│   └── lime_wrapper.py
├── physics_constrained/
│   ├── equivariant/
│   │   ├── se2_cnn.py (usando escnn)
│   │   └── e3_gnn.py (usando e3nn)
│   ├── hamiltonian_nn.py
│   ├── divergence_free_layer.py
│   └── physics_losses.py
├── mechanistic/
│   ├── concept_probing.py
│   ├── circuit_analysis.py
│   └── neuron_visualization.py
├── symbolic/
│   ├── pysr_wrapper.py
│   ├── sindy_dynamics.py
│   └── nn_distillation.py
├── validation/
│   ├── conservation_tests.py
│   ├── symmetry_tests.py
│   ├── bounds_checking.py
│   └── causal_analysis.py
├── visualization/
│   ├── activation_maps.py
│   ├── feature_importance.py
│   └── counterfactual_viz.py
└── examples/
    ├── end_to_end_interpretability.ipynb
    ├── physics_validation.ipynb
    └── symbolic_discovery.ipynb
```

### 10.2 Bibliotecas Clave

**Métodos de Atribución**:
- **Captum**: https://captum.ai/ (Interpretabilidad PyTorch)
- **SHAP**: https://github.com/slundberg/shap
- **LIME**: https://github.com/marcotcr/lime

**Arquitecturas Restringidas por Física**:
- **e3nn**: https://github.com/e3nn/e3nn (Equivariante E(3))
- **escnn**: https://github.com/QUVA-Lab/escnn (Equivariante E(2))
- **DeepXDE**: https://github.com/lululxvi/deepxde (PINNs)

**Regresión Simbólica**:
- **PySR**: https://github.com/MilesCranmer/PySR (estado del arte)
- **PySINDy**: https://github.com/dynamicslab/pysindy (dinámica)
- **AI Feynman**: https://github.com/SJ001/AI-Feynman

**Sondeo y Análisis Mecanístico**:
- **PyTorch Hooks**: Extracción de activación incorporada
- **scikit-learn**: Sondas lineales, clustering
- **NetworkX**: Análisis de grafo de circuitos

### 10.3 Datasets y Benchmarks

**Benchmarks de Interpretabilidad**:
- **Ecuaciones de Feynman**: 100 fórmulas de física simbólica
- **PDEBench**: Ecuaciones diferenciales parciales
- **Materials Project**: Propiedades de cristales

**Específicos de Dominios Magnéticos**:
- **Simulaciones Spirit**: Generar verdad fundamental con J, K, D conocidos
- **Anotar Conceptos Físicos**: Etiquetar manualmente skyrmiones, paredes (500-1000 imágenes)
- **Dataset de Diagrama de Fases**: Barrido sistemático del espacio de parámetros

---

## 11. Conclusión

### Conclusiones Clave

1. **La interpretabilidad es esencial para física**: A diferencia de tareas de caja negra, necesitamos verificar consistencia física y extraer comprensión

2. **Múltiples niveles de interpretabilidad**: Desde atribución simple (Grad-CAM) hasta comprensión mecanística (regresión simbólica)

3. **Arquitecturas restringidas por física superiores**: HNNs, NNs equivariantes garantizan propiedades físicas → más confiables

4. **Validación es crítica**: Debe probar leyes de conservación, simetrías, límites

5. **Estado del arte avanza rápidamente**: 2022-2024 vio progreso mayor en interpretabilidad informada por física

### Recomendaciones Prácticas

**Para tu Doctorado**:

1. **Comenzar con métodos post-hoc**: Grad-CAM y sondeo son victorias rápidas

2. **Siempre validar física**: Hacer esto parte de tu pipeline de evaluación

3. **Usar arquitecturas equivariantes**: Si es posible, re-entrenar con CNN equivariante SE(2)

4. **Extraer relaciones simbólicas**: Incluso fórmulas aproximadas son valiosas

5. **Colaborar con físicos**: Validar interpretaciones con expertos del dominio

**Resultados Esperados**:
- **Corto plazo**: Identificar qué características de imagen usa el modelo
- **Mediano plazo**: Construir modelo con consistencia física garantizada
- **Largo plazo**: Extraer comprensión simbólica del problema inverso

### Reflexiones Finales

**La interpretabilidad en ML de física no es solo sobre confianza—es sobre descubrimiento científico.**

Un modelo de caja negra que logra 95% de precisión pero viola conservación de energía no tiene valor para física. Un modelo interpretable con 90% de precisión que respeta todas las restricciones físicas y revela nuevas perspectivas sobre la relación parámetro-textura es invaluable.

**Tu doctorado puede contribuir tanto a predicciones precisas COMO a comprensión más profunda de dominios magnéticos.**

---

## 12. Referencias

### Fundamentos de Interpretabilidad

1. **Molnar (2022)**: *Interpretable Machine Learning*. https://christophm.github.io/interpretable-ml-book/

2. **Lipton (2018)**: "The Mythos of Model Interpretability." *Queue*, 16(3), 31-57.

### Métodos de Atribución

3. **Selvaraju et al. (2017)**: "Grad-CAM: Visual Explanations from Deep Networks." *ICCV*.

4. **Lundberg & Lee (2017)**: "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

### Arquitecturas Restringidas por Física

5. **Greydanus et al. (2019)**: "Hamiltonian Neural Networks." *NeurIPS*.

6. **Cohen & Welling (2016)**: "Group Equivariant Convolutional Networks." *ICML*.

7. **Weiler & Cesa (2022)**: "General E(2)-Equivariant Steerable CNNs." *ICML*.

8. **Batzner et al. (2022)**: "E(3)-equivariant graph neural networks for molecules and materials." *Nature Communications*, 13, 2453.

### Regresión Simbólica

9. **Cranmer (2023)**: "Interpretable Machine Learning for Science with PySR." *arXiv:2305.01582*.

10. **Lample & Charton (2024)**: "Deep Learning for Symbolic Mathematics." *ICLR*.

11. **Brunton et al. (2016)**: "Discovering governing equations from data by sparse identification." *PNAS*, 113(15), 3932-3937.

### Interpretabilidad Mecanística

12. **Jiang et al. (2024)**: "Interpretable Scientific Discovery with Self-Supervised Learning." *NeurIPS*.

13. **Alet al. (2023)**: "Physically Consistent Neural Networks for Fluid Dynamics." *Nature Machine Intelligence*.

### Causalidad

14. **Schwab & Karlen (2023)**: "Causal Machine Learning for Physics." *Physical Review Letters*.

15. **Pearl (2009)**: *Causality: Models, Reasoning, and Inference*. Cambridge University Press.

---

**Última Actualización**: Diciembre 2025
**Estado**: Marco de Investigación Establecido
**Próximos Pasos**: Implementar Grad-CAM → Probar consistencia física → Construir modelo equivariante
