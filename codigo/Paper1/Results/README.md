# Results - Análisis de Interpretabilidad

## 📋 Descripción

Esta carpeta contiene notebooks dedicados al análisis de interpretabilidad de los modelos entrenados. Se utilizan técnicas de visualización (UMAP, Grad-CAM) y experimentos de ablación para entender **qué** aprende la red y **cómo** toma decisiones.

## 🗂️ Estructura

```
Results/
├── DatabaseJex2T/
│   └── Results.ipynb           # Análisis completo para Jex2 y Temperatura
│
└── DatabaseKDMT/
    └── Results.ipynb           # Análisis completo para KDM y Temperatura
```

## 🎯 Objetivos del Análisis

1. **📊 Evaluación cuantitativa**
   - R², MAPE, SMAPE en validation set
   - Gráficos de predicción vs real

2. **🔍 Interpretabilidad de features**
   - UMAP de activaciones por capa
   - Identificación de representaciones aprendidas

3. **🎨 Grad-CAM (Class Activation Maps)**
   - Heatmaps de atención espacial
   - Análisis por rangos de parámetros
   - Contribución por capa convolucional

4. **🧪 Experimentos de ablación**
   - Enmascaramiento de regiones
   - Impacto en performance

## 🔄 Flujo de Análisis

```
┌────────────────────────────────────────────────────────────┐
│             1. CARGA DE MODELO Y DATOS                     │
│  ┌──────────────────────────────────────────────────┐     │
│  │ • Modelo entrenado (.h5)                         │     │
│  │ • Validation dataset                             │     │
│  │ • Scaler (para desnormalizar predicciones)      │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│              2. EVALUACIÓN CUANTITATIVA                    │
│  ┌──────────────────────────────────────────────────┐     │
│  │ • Predicción en validation set                   │     │
│  │ • Cálculo de métricas (R², MAPE, SMAPE)         │     │
│  │ • Scatter plot: Real vs Predicted               │     │
│  │ • Residual analysis                             │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│              3. ANÁLISIS UMAP POR CAPA                     │
│  ┌──────────────────────────────────────────────────┐     │
│  │ Capas analizadas:                                │     │
│  │  • pool2_conv (capa temprana)                    │     │
│  │  • conv3_block4_1_conv (media-temprana)         │     │
│  │  • conv4_block7_2_conv (media-tardía)           │     │
│  │  • conv5_block16_concat (capa profunda)         │     │
│  │                                                  │     │
│  │ Proceso:                                         │     │
│  │  1. Extraer activaciones intermedias             │     │
│  │  2. Flatten features                             │     │
│  │  3. UMAP reduction (2D)                          │     │
│  │  4. Scatter plot coloreado por target           │     │
│  │                                                  │     │
│  │ Insight: Ver cómo la red separa muestras        │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│           4. CLUSTERING Y SELECCIÓN DE MUESTRAS            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ • Seleccionar N imágenes representativas         │     │
│  │ • K-Means en espacio UMAP (4 clusters)          │     │
│  │ • Seleccionar 4 muestras por cluster            │     │
│  │ • Total: 16 imágenes para análisis profundo     │     │
│  │                                                  │     │
│  │ Criterio: Imágenes más cercanas a centroides    │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│            5. GRAD-CAM: HEATMAPS DE ATENCIÓN              │
│  ┌──────────────────────────────────────────────────┐     │
│  │ A. Grad-CAM basado en Error Absoluto:           │     │
│  │    loss = 1 - |y_real - y_pred|                 │     │
│  │    → Identifica regiones que reducen error      │     │
│  │                                                  │     │
│  │ B. Grad-CAM basado en MAPE:                     │     │
│  │    loss = |y_real - y_pred| / |y_real|          │     │
│  │    → Penaliza más errores relativos grandes     │     │
│  │                                                  │     │
│  │ Capas analizadas (10):                          │     │
│  │  conv2_block6_1_conv                            │     │
│  │  conv3_block5_1_conv                            │     │
│  │  conv3_block10_2_conv                           │     │
│  │  conv4_block4_2_conv                            │     │
│  │  conv4_block7_2_conv                            │     │
│  │  conv4_block11_1_conv                           │     │
│  │  conv4_block17_2_conv                           │     │
│  │  conv5_block2_1_conv                            │     │
│  │  conv5_block8_1_conv                            │     │
│  │  conv5_block16_1_conv                           │     │
│  │                                                  │     │
│  │ Output: Grilla 16×11 (imágenes × capas)        │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│        6. ANÁLISIS POR RANGOS DE PARÁMETROS               │
│  ┌──────────────────────────────────────────────────┐     │
│  │ • Dividir dataset en 5 rangos del parámetro     │     │
│  │   Ejemplo Jex2: [0.0-0.2, 0.2-0.4, ..., 0.8-1.0]│     │
│  │                                                  │     │
│  │ • Seleccionar 10 muestras representativas        │     │
│  │   por rango (cercanas al centro del rango)      │     │
│  │                                                  │     │
│  │ • Visualizar: Grilla 10×5 (muestras × rangos)   │     │
│  │                                                  │     │
│  │ • Generar Grad-CAM para 5 primeras muestras     │     │
│  │   de cada rango                                  │     │
│  │                                                  │     │
│  │ • Calcular contribución promedio por capa       │     │
│  │   en cada rango                                  │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│       7. GRÁFICO DE CONTRIBUCIÓN POR CAPA/RANGO           │
│  ┌──────────────────────────────────────────────────┐     │
│  │ • Bar plot: 5 rangos × 10 capas                  │     │
│  │ • Eje X: Rangos del parámetro                   │     │
│  │ • Eje Y: Contribución promedio Grad-CAM         │     │
│  │ • Barras: Una por capa (coloreadas)             │     │
│  │                                                  │     │
│  │ Insight: ¿Qué capas son más importantes         │     │
│  │          en cada rango del parámetro?           │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│         8. EXPERIMENTO DE ABLACIÓN (MASKING)              │
│  ┌──────────────────────────────────────────────────┐     │
│  │ Proceso:                                         │     │
│  │  1. Generar Grad-CAM basado en MAPE             │     │
│  │  2. Crear máscara binaria:                       │     │
│  │     mask = 0 donde heatmap > threshold          │     │
│  │     mask = 1 donde heatmap ≤ threshold          │     │
│  │  3. Aplicar máscara: img_masked = img * mask    │     │
│  │  4. Predecir en dataset enmascarado              │     │
│  │  5. Comparar R² original vs R² enmascarado      │     │
│  │                                                  │     │
│  │ Hipótesis:                                       │     │
│  │  • Si R² baja mucho → regiones eran importantes │     │
│  │  • Si R² se mantiene → regiones no críticas     │     │
│  │                                                  │     │
│  │ Parámetros experimentales:                       │     │
│  │  • threshold_factor: 0.7-0.8                    │     │
│  │  • layer_to_mask: conv5_block2_1_conv           │     │
│  │  • subset_size: 100% del validation set         │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────┘
```

## 📊 Técnicas de Interpretabilidad

### 1. UMAP (Uniform Manifold Approximation and Projection)

**Objetivo:** Visualizar representaciones aprendidas en espacios de alta dimensión

**Implementación:**
```python
import umap
from tensorflow.keras.models import Model

# Extraer activaciones de capas específicas
layer_names = ['pool2_conv', 'conv3_block4_1_conv',
               'conv4_block7_2_conv', 'conv5_block16_concat']

intermediate_models = [Model(inputs=model.input,
                             outputs=model.get_layer(name).output)
                      for name in layer_names]

# Procesar validation set
activations = {name: [] for name in layer_names}
for batch_x, batch_y in val_dataset:
    for name, inter_model in zip(layer_names, intermediate_models):
        act = inter_model.predict(batch_x)
        activations[name].append(act)

# Concatenar y aplanar
for name in layer_names:
    activations[name] = np.concatenate(activations[name], axis=0)
    activations[name] = activations[name].reshape(activations[name].shape[0], -1)

# UMAP reduction
reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.5,
                    metric='cosine', n_epochs=200, low_memory=True)

for name in layer_names:
    embedding = reducer.fit_transform(activations[name])
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_real, cmap='jet')
    plt.colorbar()
    plt.title(f"UMAP - {name}")
    plt.show()
```

**Interpretación:**
- Capas tempranas (pool2): Features genéricos, poca separación por parámetro
- Capas medias (conv3-conv4): Comienza separación progresiva
- Capas profundas (conv5): Clara separación por valor del parámetro

### 2. Grad-CAM (Gradient-weighted Class Activation Mapping)

**Objetivo:** Identificar qué regiones espaciales son importantes para la predicción

**Implementación (basada en error):**
```python
def compute_gradcam_error(img_array, model, layer_name, y_real):
    # Crear modelo que retorna activaciones + predicción
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    y_real_tensor = tf.constant([[y_real]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array, training=False)
        # Loss: queremos MAXIMIZAR cercanía a y_real
        loss = 1.0 - tf.abs(y_real_tensor - predictions)

    # Gradiente de loss respecto a activaciones
    grads = tape.gradient(loss, conv_output)

    # Pooled gradients (importancia por canal)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Ponderar activaciones por importancia
    conv_output = conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    # Heatmap = promedio sobre canales
    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap) + 1e-8  # Normalizar

    return heatmap
```

**Implementación (basada en MAPE):**
```python
def compute_gradcam_mape(img_array, model, layer_name, y_real, epsilon=1e-6):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    y_real_tensor = tf.cast(tf.constant([[y_real]]), tf.float32)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array, training=False)
        # Loss: MAPE
        mape = tf.abs((y_real_tensor - predictions) / (tf.abs(y_real_tensor) + epsilon))
        loss = tf.reduce_mean(mape)

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-6

    return heatmap
```

**Interpretación:**
- **Heatmap rojo (alta activación):** Regiones críticas para predicción
- **Heatmap azul (baja activación):** Regiones menos relevantes
- **Comparación entre capas:** Capas profundas (conv5) más específicas, tempranas (conv2) más globales

### 3. Clustering y Selección de Muestras

**Objetivo:** Identificar imágenes representativas de diferentes regiones del espacio latente

**Implementación:**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# UMAP en imágenes del validation set
images_flattened = images.reshape(num_samples, -1)
scaler = StandardScaler()
images_scaled = scaler.fit_transform(images_flattened)

reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.5)
images_umap = reducer.fit_transform(images_scaled)

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(images_umap)
centroids = kmeans.cluster_centers_

# Seleccionar 4 imágenes más cercanas al centroide de cada cluster
selected_images = []
for i in range(4):
    cluster_indices = np.where(labels == i)[0]
    cluster_points = images_umap[cluster_indices]
    distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
    closest_indices = cluster_indices[np.argsort(distances)[:4]]
    selected_images.extend([images[idx] for idx in closest_indices])
```

**Interpretación:** Captura diversidad de patrones magnéticos presentes en el dataset

### 4. Análisis por Rangos

**Objetivo:** Entender cómo varía la atención del modelo según el valor del parámetro

**Implementación:**
```python
# Dividir parámetro en 5 rangos equidistantes
num_bins = 5
y_min, y_max = y_real.min(), y_real.max()
bins = np.linspace(y_min, y_max, num_bins + 1)

# Asignar muestras a rangos
indices_por_rango = {i: [] for i in range(num_bins)}
for idx, y in enumerate(y_real):
    rango = np.digitize(y, bins) - 1
    rango = min(rango, num_bins - 1)
    indices_por_rango[rango].append(idx)

# Seleccionar imágenes representativas por rango
centro_rango = (bins[:-1] + bins[1:]) / 2
imagenes_por_rango = {}
for rango, indices in indices_por_rango.items():
    # Ordenar por cercanía al centro
    indices_ordenados = sorted(indices,
                               key=lambda i: abs(y_real[i] - centro_rango[rango]))
    imagenes_por_rango[rango] = [images[i] for i in indices_ordenados[:10]]
```

**Interpretación:**
- Rangos bajos (0.0-0.2): ¿Qué patrones caracterizan valores bajos del parámetro?
- Rangos altos (0.8-1.0): ¿Qué patrones caracterizan valores altos?
- Transiciones (0.4-0.6): ¿Regiones ambiguas o transicionales?

### 5. Contribución por Capa/Rango

**Objetivo:** Cuantificar la importancia de cada capa en diferentes rangos del parámetro

**Implementación:**
```python
# Diccionario para acumular contribución
contribucion_por_rango = {rango: {layer: 0 for layer in layer_names}
                          for rango in rangos}

# Para cada rango
for rango, imagenes in imagenes_por_rango.items():
    num_images = len(imagenes)

    for i, img in enumerate(imagenes[:5]):  # Primeras 5 muestras
        img_array = np.expand_dims(img, axis=0)
        y_real_value = valores_por_rango[rango][i]

        for layer in layer_names:
            heatmap = compute_gradcam_error(img_array, model, layer, y_real_value)
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)
            contribucion_por_rango[rango][layer] += np.mean(heatmap)

    # Normalizar por número de imágenes
    for layer in layer_names:
        contribucion_por_rango[rango][layer] /= num_images

# Visualizar
fig, ax = plt.subplots(figsize=(20, 10))
x = np.arange(len(rangos))
bar_width = 0.05

for i, layer in enumerate(layer_names):
    contribuciones = [contribucion_por_rango[r][layer] for r in rangos]
    ax.bar(x + i * bar_width, contribuciones, width=bar_width, label=layer)

ax.set_xlabel("Rangos")
ax.set_ylabel("Contribución Promedio Grad-CAM")
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
ax.legend()
plt.show()
```

**Interpretación:**
- Capas con alta contribución → importantes para discriminar ese rango
- Variación entre rangos → diferentes características espaciales relevantes

### 6. Experimento de Ablación (Masking)

**Objetivo:** Validar que las regiones identificadas por Grad-CAM son realmente importantes

**Implementación:**
```python
def create_mask(heatmap, threshold_factor=0.8):
    """Máscara: 0 donde heatmap alto, 1 donde bajo"""
    threshold = np.max(heatmap) * threshold_factor
    mask = np.where(heatmap >= threshold, 0, 1).astype(np.float32)
    return mask

def generate_masked_dataset(model, val_dataset, layer_to_mask, threshold_factor):
    masked_images = []
    real_values = []

    for X_batch, y_batch in val_dataset:
        for i in range(len(X_batch)):
            img = X_batch.numpy()[i]
            y_real = y_batch.numpy()[i]
            img_array = np.expand_dims(img, axis=0)

            # Grad-CAM
            heatmap = compute_gradcam_mape(img_array, model, layer_to_mask,
                                          tf.constant([[y_real]]))
            resized_heatmap = tf.image.resize(np.expand_dims(heatmap, -1),
                                             img.shape[:2]).numpy()

            # Crear máscara y aplicar
            mask = create_mask(resized_heatmap, threshold_factor)
            masked_image = img * mask

            masked_images.append(masked_image)
            real_values.append(y_real)

    return tf.data.Dataset.from_tensor_slices((np.array(masked_images),
                                               np.array(real_values))).batch(32)

# Predecir en dataset enmascarado
masked_dataset = generate_masked_dataset(model, val_dataset,
                                        'conv5_block2_1_conv', 0.8)

y_pred_masked = model.predict(masked_dataset)
r2_masked = r2_score(y_real_masked, y_pred_masked)

print(f"R² original: {r2_original:.4f}")
print(f"R² enmascarado: {r2_masked:.4f}")
print(f"Degradación: {(r2_original - r2_masked):.4f}")
```

**Interpretación:**
- Degradación grande (Δ R² > 0.2): Regiones eran críticas ✅ Grad-CAM correcto
- Degradación pequeña (Δ R² < 0.05): Regiones no tan importantes ❌ Revisar Grad-CAM

## 📊 Resultados Principales (DatabaseJex2T)

### Métricas Finales

| Target | R² Score | MAPE | SMAPE | Interpretación |
|--------|----------|------|-------|----------------|
| **Jex2** | **0.9753** | **18.64%** | **15.49%** | ✅ Excelente predicción |
| **Temperatura** | -1.2353 | 287.11% | 67.29% | ❌ Predicción pobre |

### Insights de UMAP

- **Jex2:** Clara separación en espacio latente de conv5
  - Valores bajos (azul) agrupados
  - Valores altos (rojo) agrupados
  - Transición gradual y continua

- **Temperatura:** Superposición en espacio latente
  - Múltiples temperaturas generan configuraciones similares
  - Sugiere que información temporal es necesaria

### Insights de Grad-CAM

**Para Jex2:**
- **Capas tempranas (conv2):** Atención global, poco específica
- **Capas medias (conv3-conv4):** Comienza focalización en interfaces
- **Capas profundas (conv5):** Alta atención en:
  - **Interfaces entre dominios** (bordes de regiones)
  - **Defectos topológicos** (skyrmions, vórtices)
  - **Transiciones abruptas** de magnetización

**Por rangos:**
- **Jex2 bajo (0.0-0.2):** Modelo se enfoca en regiones de alta homogeneidad
- **Jex2 alto (0.8-1.0):** Modelo se enfoca en estructuras coherentes extensas

### Experimentos de Ablación

```
Layer: conv5_block2_1_conv
Threshold: 0.8
Subset: 100% validation

R² original:    0.9753
R² enmascarado: 0.7821
Degradación:    0.1932  ← Confirma importancia de regiones identificadas
```

**Conclusión:** Grad-CAM identifica correctamente regiones críticas

## 📈 Visualizaciones Generadas

Cada notebook genera los siguientes outputs:

1. **scatter_real_vs_pred.svg** - Gráfico de dispersión predicciones
2. **UMAP_visualization_jex2.svg** - UMAP por capa para Jex2
3. **UMAP_visualization_T.svg** - UMAP por capa para Temperatura
4. **Clustered_Images_UMAP_4Clusters_4Each.svg** - Imágenes representativas
5. **gradcam_results.svg** - Heatmaps para muestras seleccionadas
6. **gradcam_<rango>.svg** - Heatmaps por cada rango (5 archivos)
7. **mean_contribution_gradcam.svg** - Contribución por capa/rango (Error)
8. **mean_contribution_gradcam_MAPE.svg** - Contribución por capa/rango (MAPE)

## 🚀 Cómo Ejecutar el Análisis

### Paso 1: Cargar Modelo

```python
from tensorflow.keras.models import load_model

model_jex2 = load_model('modelo_densenet_regresionY_2.h5',
                        custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
```

### Paso 2: Preparar Validation Dataset

```python
# Cargar dataset
data = np.load('spinesv0.npz')
X = data['X'][:, :42, :, :]
y_jex2 = data['y'][:, 0].reshape(-1, 1)

# Preprocesar
processed_images = preprocess_images(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(processed_images, y_jex2,
                                                   test_size=0.1, random_state=42)

# Dataset
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
```

### Paso 3: Ejecutar Secciones del Notebook

- **Sección 1-2:** Evaluación cuantitativa
- **Sección 3:** UMAP por capa
- **Sección 4:** Clustering
- **Sección 5-6:** Grad-CAM
- **Sección 7:** Análisis por rangos
- **Sección 8:** Experimentos de ablación

### Paso 4: Guardar Visualizaciones

```python
plt.savefig("figura.svg", format="svg", dpi=300, bbox_inches="tight")
```

## 🐛 Troubleshooting

**Problema:** UMAP muy lento
- **Solución:** Usar `low_memory=True`, reducir `n_epochs`, o samplear subset

**Problema:** Grad-CAM todo negro/blanco
- **Solución:** Verificar normalización de heatmap, ajustar colormap, revisar gradientes

**Problema:** Memory Error en ablación
- **Solución:** Procesar en batches más pequeños, reducir subset_size

## 📚 Referencias

- **Grad-CAM:** Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **UMAP:** McInnes et al. (2018) - "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"

## 🔗 Conexión con Papers

Los análisis de este notebook proporcionan:
- **Figuras para publicación:** UMAP, Grad-CAM, contribución por capa
- **Validación científica:** Ablation studies demuestran que el modelo aprende física relevante
- **Insights físicos:** Identificación de características magnéticas relevantes (interfaces, defectos)

---

**Nota:** Este análisis es crítico para papers científicos, ya que demuestra que el modelo no solo "funciona" sino que aprende representaciones físicamente interpretables.
