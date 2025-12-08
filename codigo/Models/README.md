# Models - Entrenamiento de Redes Neuronales

## 📋 Descripción

Esta carpeta contiene los notebooks de entrenamiento de diferentes arquitecturas de redes neuronales convolucionales (CNNs) para la predicción de parámetros hamilton

ianos a partir de configuraciones de spin magnéticas.

## 🗂️ Estructura

```
Models/
├── DatabaseJex2T/              # Modelos para predecir Jex2 y Temperatura
│   ├── DenseNetFinal.ipynb      ✅ MEJOR MODELO (R²=0.9753)
│   ├── EfficienNetFinal.ipynb
│   └── ResNetFinal.ipynb
│
└── DatabaseKDMT/               # Modelos para predecir KDM y Temperatura
    ├── DenseNet121KDM.ipynb
    ├── EfficientNetB2KDM.ipynb
    ├── EfficientNetKDM.ipynb
    ├── InceptionNetKDM.ipynb
    └── ResNetKDM.ipynb
```

## 🏆 Comparación de Arquitecturas

### DatabaseJex2T (Predicción de Jex2)

| Modelo | R² Score | MAPE | SMAPE | Parámetros | Tiempo/Época |
|--------|----------|------|-------|------------|--------------|
| **DenseNet121** ✅ | **0.9753** | **18.64%** | **15.49%** | ~7M | ~3 min |
| ResNet50 | ~0.94 | ~22% | ~18% | ~23M | ~2 min |
| EfficientNetB2 | ~0.95 | ~20% | ~17% | ~7.8M | ~4 min |

**Ganador:** DenseNet121 por mejor R² y balance eficiencia/performance

### DatabaseKDMT (Predicción de KDM)

| Modelo | Estado | Observaciones |
|--------|--------|---------------|
| DenseNet121KDM | En progreso | Arquitectura prometedora |
| EfficientNetB2KDM | En progreso | Balance eficiencia/performance |
| EfficientNetKDM | En progreso | Versión B7 más pesada |
| InceptionNetKDM | En progreso | Multi-scale features |
| ResNetKDM | En progreso | Baseline con residuals |

## 🔄 Pipeline de Entrenamiento

```
┌────────────────────────────────────────────────────────────┐
│                 1. CARGA DE DATOS                          │
│  ┌──────────────────────────────────────────────────┐     │
│  │ • Descarga dataset desde Kaggle/Drive            │     │
│  │ • Carga archivo .npz                             │     │
│  │ • Extracción: X (imágenes), y (parámetros)      │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│              2. PREPROCESAMIENTO                           │
│  ┌──────────────────────────────────────────────────┐     │
│  │ • Resize: 42×42 → 224×224 (TF resize)           │     │
│  │ • Conversión: Grayscale → RGB (3 canales)       │     │
│  │ • Normalización imágenes (global min/max)       │     │
│  │ • MinMaxScaler en targets (y)                   │     │
│  │ • Train/Val split: 90/10 (stratified)          │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│              3. CONSTRUCCIÓN DEL MODELO                    │
│  ┌──────────────────────────────────────────────────┐     │
│  │ Base Model (Pre-trained on ImageNet)            │     │
│  │  ├─ DenseNet121 / ResNet50 / EfficientNet      │     │
│  │  ├─ Freeze initial layers                       │     │
│  │  └─ Trainable: últimas N capas                  │     │
│  │                                                  │     │
│  │ Custom Head:                                     │     │
│  │  ├─ GlobalAveragePooling2D()                    │     │
│  │  ├─ Dense(256, relu, dropout=0.3)              │     │
│  │  ├─ Dense(128, relu, dropout=0.2)              │     │
│  │  └─ Dense(1, linear) ← Regresión               │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│                  4. ENTRENAMIENTO                          │
│  ┌──────────────────────────────────────────────────┐     │
│  │ Optimizer: Adam (lr=1e-4, decay=1e-6)           │     │
│  │ Loss: Mean Squared Error (MSE)                   │     │
│  │ Metrics: MAE, MAPE                              │     │
│  │                                                  │     │
│  │ Callbacks:                                       │     │
│  │  • ModelCheckpoint (save_best_only=True)        │     │
│  │  • EarlyStopping (patience=10, monitor='val_loss')│   │
│  │  • ReduceLROnPlateau (factor=0.5, patience=5)   │     │
│  │                                                  │     │
│  │ Batch size: 32                                   │     │
│  │ Epochs: 50-100 (early stopping)                 │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│                 5. EVALUACIÓN                              │
│  ┌──────────────────────────────────────────────────┐     │
│  │ Métricas en Validation Set:                     │     │
│  │  • R² Score                                      │     │
│  │  • MAPE (masked, epsilon=1e-8)                  │     │
│  │  • SMAPE                                         │     │
│  │                                                  │     │
│  │ Visualizaciones:                                 │     │
│  │  • Scatter: y_real vs y_pred                    │     │
│  │  • Learning curves (loss vs epochs)             │     │
│  │  • Residual plots                               │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│                 6. GUARDADO                                │
│  ┌──────────────────────────────────────────────────┐     │
│  │ • Modelo completo: .h5 format                   │     │
│  │ • Pesos: checkpoint.weights.h5                   │     │
│  │ • Historial: history.json                       │     │
│  │ • Scaler: scaler.pkl (para inferencia)          │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────┘
```

## 🏗️ Arquitecturas Implementadas

### 1. DenseNet121 ⭐ MEJOR MODELO

**Arquitectura:**
```
Input (224, 224, 3)
    ↓
DenseNet121 Base (ImageNet weights)
├─ Conv2D (7×7, stride=2)
├─ MaxPooling (3×3, stride=2)
├─ Dense Block 1 (6 layers)
├─ Transition Layer 1
├─ Dense Block 2 (12 layers)
├─ Transition Layer 2
├─ Dense Block 3 (24 layers)
├─ Transition Layer 3
└─ Dense Block 4 (16 layers)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) + Dropout(0.3)
    ↓
Dense(128, relu) + Dropout(0.2)
    ↓
Dense(1, linear) ← Output
```

**Ventajas:**
- ✅ Conexiones densas → mejor flujo de gradientes
- ✅ Feature reuse → menos parámetros (~7M vs ~23M ResNet)
- ✅ Mejor generalización en datasets pequeños/medianos
- ✅ R² = 0.9753 en Jex2

**Hiperparámetros:**
```python
base_model = DenseNet121(include_top=False,
                        weights='imagenet',
                        input_shape=(224, 224, 3))

# Freeze primeras capas
for layer in base_model.layers[:-50]:
    layer.trainable = False

optimizer = Adam(learning_rate=1e-4, decay=1e-6)
loss = 'mse'
```

### 2. ResNet50

**Arquitectura:**
```
Input (224, 224, 3)
    ↓
ResNet50 Base (ImageNet weights)
├─ Conv2D (7×7, stride=2)
├─ MaxPooling (3×3, stride=2)
├─ Conv Block 1 (3 layers) + Skip
├─ Conv Block 2 (4 layers) + Skip
├─ Conv Block 3 (6 layers) + Skip
├─ Conv Block 4 (3 layers) + Skip
└─ Conv Block 5 (3 layers) + Skip
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, relu) + Dropout(0.4)
    ↓
Dense(256, relu) + Dropout(0.3)
    ↓
Dense(1, linear) ← Output
```

**Ventajas:**
- ✅ Residual connections → entrenamientos muy profundos
- ✅ Arquitectura muy probada y estable
- ✅ Baseline confiable

**Desventajas:**
- ❌ Más parámetros (~23M)
- ❌ Ligeramente inferior a DenseNet en este problema

### 3. EfficientNetB2/B7

**Arquitectura:**
```
Input (224, 224, 3)
    ↓
EfficientNet Base (Compound Scaling)
├─ Stem: Conv2D (3×3)
├─ MBConv Blocks (depth, width, resolution scaling)
│   ├─ Depthwise Conv
│   ├─ Squeeze-Excitation
│   └─ Skip connection
└─ Head: Conv2D (1×1)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) + Dropout(0.3)
    ↓
Dense(1, linear) ← Output
```

**Ventajas:**
- ✅ Compound scaling balanceado
- ✅ Eficiente en parámetros y FLOPs
- ✅ State-of-the-art en ImageNet

**Desventajas:**
- ❌ B7 es muy pesado para este problema
- ❌ B2 competitivo pero no supera DenseNet

### 4. InceptionNetV3

**Arquitectura:**
```
Input (224, 224, 3)
    ↓
InceptionNet Base
├─ Conv2D inicial
├─ Inception Module A (multi-scale 1×1, 3×3, 5×5)
├─ Reduction A
├─ Inception Module B
├─ Reduction B
└─ Inception Module C
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, relu) + Dropout(0.4)
    ↓
Dense(256, relu) + Dropout(0.3)
    ↓
Dense(1, linear) ← Output
```

**Ventajas:**
- ✅ Multi-scale feature extraction
- ✅ Captura patrones a diferentes escalas

**Aplicación:** Útil si dominios magnéticos tienen estructuras multi-escala

## 📊 Métricas de Evaluación

### R² Score (Coeficiente de Determinación)
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_real, y_pred)
# Interpretación:
# 1.0 = Predicción perfecta
# 0.0 = Modelo no mejor que predecir la media
# <0.0 = Modelo peor que predecir la media
```

### MAPE (Mean Absolute Percentage Error)
```python
def masked_mape(y_true, y_pred, epsilon=1e-8):
    """MAPE con máscara para evitar división por cero"""
    mask = np.abs(y_true) > epsilon
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
```

### SMAPE (Symmetric Mean Absolute Percentage Error)
```python
def smape(y_true, y_pred, epsilon=1e-8):
    """SMAPE simétrico, más robusto que MAPE"""
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100
```

## 🎛️ Hiperparámetros Clave

### Transfer Learning Strategy

```python
# Estrategia 1: Freeze + Fine-tuning
base_model.trainable = False  # Entrenar solo head
# ... entrenar 10 epochs ...
base_model.trainable = True   # Unfreeze
for layer in base_model.layers[:-50]:
    layer.trainable = False   # Freeze solo primeras capas
# ... entrenar 40 epochs más ...

# Estrategia 2: Freeze parcial desde inicio
for layer in base_model.layers[:-50]:
    layer.trainable = False
# ... entrenar 50 epochs ...
```

### Learning Rate Schedule

```python
# Opción 1: ReduceLROnPlateau
ReduceLROnPlateau(monitor='val_loss',
                  factor=0.5,
                  patience=5,
                  min_lr=1e-7)

# Opción 2: Cosine Decay
tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000)
```

### Regularización

```python
# Dropout en head
Dense(256, activation='relu')
Dropout(0.3)  # 30% dropout

# L2 Regularization
Dense(256, activation='relu',
      kernel_regularizer=l2(0.01))

# Data Augmentation (si aplica)
tf.keras.layers.RandomFlip("horizontal")
tf.keras.layers.RandomRotation(0.2)
```

## 🚀 Cómo Ejecutar un Entrenamiento

### Paso 1: Preparar Entorno

```python
# En Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Instalar dependencias
!pip install -q umap-learn
```

### Paso 2: Cargar Dataset

```python
import numpy as np

file_ = '/content/drive/MyDrive/DoctoradoPaper1/DataSets/spinesv0.npz'
data = np.load(file_)

X = data['X'][:, :42, :, :]  # Imágenes
y_jex2 = data['y'][:, 0].reshape(-1, 1)  # Target Jex2
y_T = data['y'][:, 1].reshape(-1, 1)     # Target Temperatura
```

### Paso 3: Preprocesar

```python
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Resize a 224×224
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    return image

processed_images = np.array([preprocess_image(img) for img in X])

# Normalizar targets
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y_jex2)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    processed_images, y_scaled, test_size=0.1, random_state=42
)

# Crear tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
```

### Paso 4: Construir Modelo

```python
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Base model
base = DenseNet121(include_top=False, weights='imagenet',
                   input_shape=(224, 224, 3))

# Freeze
for layer in base.layers[:-50]:
    layer.trainable = False

# Head
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='linear')(x)

model = Model(inputs=base.input, outputs=output)
```

### Paso 5: Compilar y Entrenar

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

model.compile(
    optimizer=Adam(1e-4),
    loss='mse',
    metrics=['mae']
)

callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks
)
```

### Paso 6: Evaluar

```python
from sklearn.metrics import r2_score

y_pred = model.predict(X_val)
y_pred_original = scaler.inverse_transform(y_pred)
y_val_original = scaler.inverse_transform(y_val)

r2 = r2_score(y_val_original, y_pred_original)
print(f"R² Score: {r2:.4f}")

# Visualizar
plt.scatter(y_val_original, y_pred_original, alpha=0.5)
plt.plot([y_val_original.min(), y_val_original.max()],
         [y_val_original.min(), y_val_original.max()], 'k--')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.title(f'R² = {r2:.4f}')
plt.show()
```

## 🐛 Troubleshooting

**Problema:** Overfitting (train loss << val loss)
- **Solución:** Aumentar dropout, añadir L2 regularization, reducir capacidad del modelo

**Problema:** Underfitting (train loss y val loss altos)
- **Solución:** Unfreeze más capas, aumentar lr, añadir capas en head

**Problema:** Nan loss
- **Solución:** Reducir lr, verificar normalización de datos, clip gradients

**Problema:** GPU Out of Memory
- **Solución:** Reducir batch_size, usar model de menor tamaño (B0 en vez de B7)

## 📈 Mejores Prácticas

1. **✅ Siempre usar validation set** para early stopping
2. **✅ Guardar solo best model** (save_best_only=True)
3. **✅ Normalizar targets** con MinMaxScaler o StandardScaler
4. **✅ Experimentar con diferentes freezing strategies**
5. **✅ Monitorear métricas físicamente interpretables** (no solo loss)
6. **✅ Visualizar predicciones** durante entrenamiento

## 🔗 Siguiente Paso

Una vez entrenado el modelo:
→ Ir a `../Results/` para análisis de interpretabilidad

---

**Nota:** Los notebooks incluyen código completo de entrenamiento. Esta documentación resume la metodología común a todos.
