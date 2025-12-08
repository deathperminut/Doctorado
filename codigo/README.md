# Predicción de Parámetros Hamiltonianos en Dominios Magnéticos mediante Deep Learning

## 📋 Descripción General

Este proyecto investiga el uso de redes neuronales convolucionales (CNNs) profundas para predecir parámetros físicos de sistemas magnéticos (hamiltonianos) a partir de imágenes de configuraciones de spin. Se utilizan múltiples arquitecturas de deep learning (DenseNet, ResNet, EfficientNet, InceptionNet) para establecer relaciones entre patrones espaciales de magnetización y sus parámetros físicos subyacentes.

## 🎯 Objetivo

Desarrollar modelos de aprendizaje profundo capaces de:
1. Predecir parámetros del hamiltoniano (Jex2, KDM, Temperatura) a partir de imágenes de estados magnéticos
2. Identificar qué características espaciales de los dominios magnéticos son más relevantes para cada parámetro
3. Interpretar las decisiones del modelo mediante técnicas de visualización (UMAP, Grad-CAM)

## 📊 Bases de Datos

El proyecto trabaja con dos bases de datos de simulaciones magnéticas:

### 1. **DatabaseJex2T**
- **Parámetros variables:** Jex2 (interacción de intercambio) y Temperatura (T)
- **Imágenes:** Configuraciones espaciales de spin (39×39 píxeles)
- **Total muestras:** ~54,044 imágenes
- **Objetivo:** Predecir Jex2 y T a partir de patrones de magnetización

### 2. **DatabaseKDMT**
- **Parámetros variables:** KDM (anisotropía Dzyaloshinskii-Moriya) y Temperatura (T)
- **Imágenes:** Configuraciones espaciales de spin (39×39 píxeles)
- **Total muestras:** Variable según construcción
- **Objetivo:** Predecir KDM y T a partir de patrones de magnetización

## 🔄 Flujo de Trabajo

```
┌─────────────────────────────────────────────────────────────┐
│                    1. PREPROCESSING                          │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │ Archivos .dat  │────────▶│ Construction.ipynb│           │
│  │ (Simulaciones) │         └────────┬──────────┘           │
│  └────────────────┘                  │                      │
│                                      ▼                       │
│                        ┌─────────────────────────┐          │
│                        │  Imágenes de Spin       │          │
│                        │  (39×39×1 → 224×224×3)  │          │
│                        └────────┬─────────────────┘          │
│                                 │                            │
│                                 ▼                            │
│                        ┌──────────────────────┐             │
│                        │ DescriptionRescale   │             │
│                        │ • Normalización      │             │
│                        │ • Visualización UMAP │             │
│                        └──────────┬────────────┘             │
│                                   │                          │
│                                   ▼                          │
│                        ┌──────────────────────┐             │
│                        │ Dataset .npz         │             │
│                        │ (Kaggle Storage)     │             │
│                        └──────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      2. MODELS                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Arquitecturas Probadas                    │   │
│  │                                                       │   │
│  │  • DenseNet121    ✅ MEJOR MODELO (R² > 0.97)       │   │
│  │  • ResNet50                                          │   │
│  │  • EfficientNetB2/B7                                 │   │
│  │  • InceptionNetV3                                    │   │
│  │                                                       │   │
│  │  Entrenamiento:                                      │   │
│  │  ├─ Train/Val Split: 90/10                          │   │
│  │  ├─ Optimizador: Adam                               │   │
│  │  ├─ Loss: MSE                                        │   │
│  │  ├─ Transfer Learning: ImageNet weights             │   │
│  │  └─ Fine-tuning de capas finales                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    3. RESULTS                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Análisis de Interpretabilidad           │   │
│  │                                                       │   │
│  │  📊 Métricas de Performance:                         │   │
│  │     • R² Score                                       │   │
│  │     • MAPE (Mean Absolute Percentage Error)          │   │
│  │     • SMAPE (Symmetric MAPE)                         │   │
│  │                                                       │   │
│  │  🔍 Visualización de Features:                       │   │
│  │     • UMAP por capa de la red                        │   │
│  │     • Clustering de activaciones                     │   │
│  │                                                       │   │
│  │  🎨 Grad-CAM (Interpretabilidad):                    │   │
│  │     • Heatmaps de atención del modelo               │   │
│  │     • Análisis por rangos de parámetros             │   │
│  │     • Identificación de regiones críticas           │   │
│  │     • Análisis de contribución por capa              │   │
│  │                                                       │   │
│  │  📈 Experimentos de Ablación:                        │   │
│  │     • Enmascaramiento de regiones                   │   │
│  │     • Impacto en performance                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

```
codigo/
├── preprocessing/          # Construcción y preparación de datos
│   ├── Databasejex2T/
│   │   ├── Construction.ipynb          # Construye dataset desde .dat
│   │   └── DescriptionRescale.ipynb    # Análisis y normalización
│   └── DatabaseKDMT/
│       ├── Construction.ipynb
│       └── DescriptionRescale.ipynb
│
├── Models/                 # Entrenamiento de modelos
│   ├── DatabaseJex2T/
│   │   ├── DenseNetFinal.ipynb         # ✅ Mejor modelo
│   │   ├── EfficienNetFinal.ipynb
│   │   └── ResNetFinal.ipynb
│   └── DatabaseKDMT/
│       ├── DenseNet121KDM.ipynb
│       ├── EfficientNetB2KDM.ipynb
│       ├── EfficientNetKDM.ipynb
│       ├── InceptionNetKDM.ipynb
│       └── ResNetKDM.ipynb
│
└── Results/                # Análisis e interpretabilidad
    ├── DatabaseJex2T/
    │   └── Results.ipynb               # Análisis completo Jex2/T
    └── DatabaseKDMT/
        └── Results.ipynb               # Análisis completo KDM/T
```

## 🏆 Resultados Principales

### DatabaseJex2T (Parámetro Jex2)

**Modelo:** DenseNet121

| Métrica | Valor |
|---------|-------|
| **R² Score** | **0.9753** |
| MAPE | 18.64% |
| SMAPE | 15.49% |

**Interpretación:**
- El modelo captura exitosamente la relación entre patrones de spin y Jex2
- Las capas profundas (conv4, conv5) muestran mayor sensibilidad a variaciones del parámetro
- Grad-CAM revela que el modelo se enfoca en interfaces de dominios magnéticos

### DatabaseJex2T (Temperatura)

**Modelo:** DenseNet121

| Métrica | Valor |
|---------|-------|
| R² Score | -1.2353 |
| MAPE | 287.11% |
| SMAPE | 67.29% |

**Interpretación:**
- La temperatura es más difícil de predecir desde configuraciones estáticas
- Sugiere que múltiples temperaturas pueden producir configuraciones similares
- Requiere información adicional (dinámica temporal, fluctuaciones)

## 🔬 Metodología

### 1. Preprocesamiento
- **Input:** Archivos .dat con coordenadas y componentes de spin
- **Transformación:** Conversión a imágenes 2D de magnetización (componente Sz)
- **Normalización:** MinMaxScaler sobre valores de magnetización
- **Resize:** 39×39 → 224×224 (requerido por CNNs pre-entrenadas)
- **Augmentation:** Repetición en 3 canales RGB (compatibilidad ImageNet)

### 2. Arquitecturas de Modelos

**DenseNet121** (Seleccionado como mejor modelo):
- Conexiones densas entre capas → mejor flujo de gradientes
- Transfer learning desde ImageNet
- Capa final: Global Average Pooling → Dense(1) para regresión
- Parámetros entrenables: ~7M

**Otras arquitecturas probadas:**
- ResNet50: Residual connections
- EfficientNetB2/B7: Compound scaling
- InceptionNetV3: Multi-scale feature extraction

### 3. Entrenamiento
- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (lr=1e-4)
- **Batch size:** 32
- **Train/Val split:** 90/10
- **Early stopping:** Patience=10 epochs
- **Callbacks:** ModelCheckpoint, ReduceLROnPlateau

### 4. Interpretabilidad

**UMAP (Uniform Manifold Approximation and Projection):**
- Visualización de activaciones intermedias
- Análisis de separabilidad en espacio latente
- Identificación de clusters por rango de parámetros

**Grad-CAM (Gradient-weighted Class Activation Mapping):**
- Heatmaps de atención del modelo
- Análisis por capa (conv2 → conv5)
- Identificación de regiones críticas para predicción
- Contribución promedio por rango de parámetros

**Experimentos de Ablación:**
- Enmascaramiento de regiones de alta/baja atención
- Medición de impacto en R² score
- Validación de interpretaciones Grad-CAM

## 📊 Visualizaciones Clave

### 1. UMAP de Activaciones
Muestra cómo las capas de la red separan progresivamente las muestras según su valor de parámetro.

### 2. Grad-CAM Heatmaps
Revela qué regiones espaciales de la imagen son más importantes para la predicción.

### 3. Análisis por Rangos
Divide el espacio de parámetros en 5 rangos y analiza el comportamiento del modelo en cada uno.

### 4. Contribución por Capa
Gráfico de barras mostrando la importancia relativa de cada capa convolucional.

## 🛠️ Tecnologías Utilizadas

- **Python 3.10+**
- **TensorFlow/Keras 2.x** - Deep Learning
- **NumPy, Pandas** - Manipulación de datos
- **Matplotlib, Seaborn** - Visualización
- **UMAP-learn** - Reducción dimensional
- **scikit-learn** - Métricas y preprocesamiento
- **OpenCV (cv2)** - Procesamiento de imágenes
- **Google Colab** - Entrenamiento con GPU

## 🚀 Cómo Usar

### 1. Construir Base de Datos
```bash
# Ejecutar notebook de construcción
codigo/preprocessing/Databasejex2T/Construction.ipynb
```

### 2. Entrenar Modelo
```bash
# Ejecutar notebook de modelo seleccionado
codigo/Models/DatabaseJex2T/DenseNetFinal.ipynb
```

### 3. Analizar Resultados
```bash
# Ejecutar análisis de interpretabilidad
codigo/Results/DatabaseJex2T/Results.ipynb
```

## 📝 Notebooks Clave

| Notebook | Descripción | Tiempo estimado |
|----------|-------------|-----------------|
| `Construction.ipynb` | Construye dataset desde simulaciones | 15-30 min |
| `DenseNetFinal.ipynb` | Entrena modelo DenseNet | 2-4 horas (GPU) |
| `Results.ipynb` | Análisis completo de interpretabilidad | 30-60 min |

## 🔍 Próximos Pasos

- [ ] Explorar arquitecturas tipo Vision Transformer (ViT)
- [ ] Incorporar información temporal (series de configuraciones)
- [ ] Multi-task learning (predecir múltiples parámetros simultáneamente)
- [ ] Análisis de incertidumbre (Bayesian Neural Networks)
- [ ] Validación en datos experimentales reales

## 📚 Referencias

- **DenseNet:** Huang et al. (2017) - Densely Connected Convolutional Networks
- **Grad-CAM:** Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep Networks
- **UMAP:** McInnes et al. (2018) - UMAP: Uniform Manifold Approximation and Projection

## 👤 Autor

Juan Sebastián Méndez Rondón
Proyecto de Doctorado - Dominios Magnéticos y Deep Learning

---

**Última actualización:** Diciembre 2025
