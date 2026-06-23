# Preprocessing - Construcción de Bases de Datos

## 📋 Descripción

Esta carpeta contiene los notebooks para construir, procesar y analizar las bases de datos de configuraciones magnéticas. El proceso transforma datos brutos de simulaciones (archivos .dat) en datasets estructurados listos para entrenamiento de modelos de deep learning.

## 🗂️ Estructura

```
preprocessing/
├── Databasejex2T/          # Dataset con Jex2 y Temperatura variables
│   ├── Construction.ipynb          # Construcción del dataset
│   └── DescriptionRescale.ipynb    # Análisis y normalización
│
└── DatabaseKDMT/           # Dataset con KDM y Temperatura variables
    ├── Construction.ipynb
    └── DescriptionRescale.ipynb
```

## 🔄 Flujo de Procesamiento

```
┌──────────────────────┐
│   Archivos .dat      │  ◀─ Simulaciones Monte Carlo
│   (Kaggle Input)     │     37 archivos States*.dat
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│            📓 Construction.ipynb                      │
│                                                       │
│  1️⃣  Lectura de archivos .dat                        │
│      • Parámetros del hamiltoniano                   │
│      • Coordenadas espaciales (x, y)                 │
│      • Componentes de spin (Sx, Sy, Sz)              │
│                                                       │
│  2️⃣  Generación de imágenes 2D                       │
│      • Mapeo de Sz a matriz 39×39                    │
│      • Cada imagen = configuración de spin          │
│                                                       │
│  3️⃣  Extracción de parámetros                        │
│      • Nest, L, rd, So, T, Jex, Jex2, Jex3, Jex4    │
│      • Kan1, Kan2, KanS, Hex, kd, KDM               │
│                                                       │
│  4️⃣  Consolidación                                    │
│      • MS: Array de imágenes (N, 39, 39, 1)         │
│      • Parámetros: Arrays 1D por cada variable      │
│                                                       │
│  ✅ Output: ~54,044 imágenes + parámetros            │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│        📓 DescriptionRescale.ipynb                    │
│                                                       │
│  1️⃣  Análisis exploratorio                           │
│      • Distribución de parámetros                   │
│      • Estadísticas descriptivas                    │
│      • Visualización de muestras                    │
│                                                       │
│  2️⃣  UMAP Visualization                              │
│      • Reducción dimensional 2D                     │
│      • Coloreado por parámetros                     │
│      • Identificación de clusters                   │
│                                                       │
│  3️⃣  Normalización                                    │
│      • MinMaxScaler en imágenes                     │
│      • Valores en rango [0, 1]                      │
│                                                       │
│  4️⃣  Export a Kaggle                                 │
│      • Formato .npz comprimido                      │
│      • Upload automático a Kaggle Datasets          │
│                                                       │
│  ✅ Output: Dataset listo para modelos               │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│   Dataset .npz       │
│   (Kaggle Storage)   │
│   • MS (imágenes)    │
│   • Parámetros       │
└──────────────────────┘
```

## 📊 Formato de Datos

### Archivos de Entrada (.dat)

Cada archivo contiene simulaciones Monte Carlo de configuraciones de spin:

**Estructura:**
```
Línea 1: Nest  L  rd  So  T  Jex  Jex2  Jex3  Jex4  Kan1  Kan2  KanS  Hex  kd  KDM
Líneas 2-N: AtomID  x  y  Sx  Sy  Sz
Línea N+1: Nest  L  rd  So  T  Jex  Jex2  Jex3  Jex4  Kan1  Kan2  KanS  Hex  kd  KDM
Líneas N+2-2N: AtomID  x  y  Sx  Sy  Sz
...
```

**Parámetros del Hamiltoniano:**
- **Nest:** Número de estado
- **L:** Tamaño del sistema (lado)
- **rd:** Radio máximo
- **So:** Spin orbital
- **T:** Temperatura (kelvin)
- **Jex, Jex2, Jex3, Jex4:** Constantes de intercambio
- **Kan1, Kan2, KanS:** Anisotropías
- **Hex:** Campo externo
- **kd:** Constante de acoplamiento
- **KDM:** Interacción Dzyaloshinskii-Moriya

### Archivos de Salida (.npz)

**DatabaseJex2T:**
```python
{
    'MS': np.array,     # Shape: (54044, 39, 39, 1) - Imágenes de spin
    'Nest': np.array,   # Shape: (54044,) - Estado
    'L': np.array,      # Shape: (54044,) - Tamaño sistema
    'T': np.array,      # Shape: (54044,) - Temperatura ⭐ TARGET
    'Jex2': np.array,   # Shape: (54044,) - Intercambio ⭐ TARGET
    'Jex3': np.array,   # Shape: (54044,) - Intercambio
    'Jex4': np.array,   # Shape: (54044,) - Intercambio
    ... # Otros parámetros
}
```

**DatabaseKDMT:**
```python
{
    'MS': np.array,     # Shape: (N, 39, 39, 1) - Imágenes de spin
    'KDM': np.array,    # Shape: (N,) - DM interaction ⭐ TARGET
    'T': np.array,      # Shape: (N,) - Temperatura ⭐ TARGET
    ... # Otros parámetros
}
```

## 📓 Descripción de Notebooks

### 1. Construction.ipynb

**Propósito:** Construir el dataset desde archivos .dat de simulaciones

**Funciones principales:**

```python
def find_dat_files(folder_path):
    """Busca todos los archivos .dat en carpeta"""
    return glob.glob(os.path.join(folder_path, '*.dat'))

def generateImage(select_file, sample_file):
    """
    Convierte archivo .dat en imágenes y parámetros

    Parameters:
    -----------
    select_file : str
        Ruta al archivo States*.dat
    sample_file : str
        Ruta al archivo Sample.dat (coordenadas)

    Returns:
    --------
    MS : np.array (N, 39, 39, 1)
        Imágenes de componente Sz de spin
    Nest, L, rd, So, T, ... : np.array (N,)
        Parámetros del hamiltoniano
    """
    # Leer tamaño del sistema
    # Leer coordenadas de Sample.dat
    # Para cada configuración:
    #   - Leer parámetros
    #   - Leer valores de Sz
    #   - Mapear a imagen 2D
    # Retornar imágenes y parámetros
```

**Pipeline:**
1. Localizar archivos .dat
2. Filtrar archivos problemáticos
3. Procesar cada archivo con `generateImage()`
4. Concatenar resultados de todos los archivos
5. Resultado final: ~54,044 imágenes

**Output esperado:**
```
MS.shape = (54044, 39, 39, 1)
T.shape = (54044,)
Jex2.shape = (54044,)
```

### 2. DescriptionRescale.ipynb

**Propósito:** Analizar, normalizar y exportar el dataset

**Secciones:**

#### A. Análisis Exploratorio

```python
# Estadísticas de parámetros
print(f"Temperatura - Min: {T.min()}, Max: {T.max()}, Mean: {T.mean()}")
print(f"Jex2 - Min: {Jex2.min()}, Max: {Jex2.max()}, Mean: {Jex2.mean()}")

# Visualización de muestras
plt.imshow(MS[i, :, :, 0], cmap='jet')
```

#### B. UMAP Visualization

```python
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

# Reducción dimensional
reducer = UMAP(n_components=2, n_neighbors=20, min_dist=0.1)
Z = reducer.fit_transform(MS.reshape(MS.shape[0], -1))

# Visualización
plt.scatter(Z[:, 0], Z[:, 1], c=T, cmap='jet')
plt.title('UMAP 2D - Colored by Temperature')
```

**Propósito:** Verificar que parámetros diferentes generan configuraciones distinguibles

#### C. Normalización

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
MS_normalized = scaler.fit_transform(MS.reshape(MS.shape[0], -1))
MS_normalized = MS_normalized.reshape(MS.shape)
```

#### D. Export a Kaggle

```python
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# Guardar como .npz
np.savez('data.npy',
         MS=MS, Nest=Nest, L=L, T=T,
         Jex2=Jex2, Jex3=Jex3, ...)

# Configurar metadatos
metadata = {
    'title': 'Material Spinners Data Models',
    'id': 'deathperminut/material-spinners-data',
    'licenses': [{'name': 'CC0-1.0'}]
}

# Upload a Kaggle
!kaggle datasets version -p /kaggle/working -m "Updated dataset"
```

## 🎯 Objetivos del Preprocessing

1. **✅ Correcta extracción de datos**
   - Verificar integridad de archivos .dat
   - Mapeo correcto de coordenadas a píxeles
   - Preservación de todos los parámetros

2. **✅ Calidad de datos**
   - Sin valores NaN o infinitos
   - Distribuciones razonables de parámetros
   - Coherencia física de configuraciones

3. **✅ Preparación para ML**
   - Normalización adecuada
   - Formato compatible con TensorFlow/Keras
   - Datasets balanceados (si es posible)

4. **✅ Reproducibilidad**
   - Seeds fijos para splits
   - Versionado en Kaggle
   - Documentación de transformaciones

## 🔍 Checks de Calidad

Antes de proceder a entrenamiento, verificar:

```python
# 1. Shape correcto
assert MS.shape == (54044, 39, 39, 1)
assert len(T) == 54044

# 2. Valores en rango esperado
assert MS.min() >= -1 and MS.max() <= 1  # Spin -1 a +1
assert T.min() > 0  # Temperatura positiva

# 3. Sin valores faltantes
assert not np.isnan(MS).any()
assert not np.isnan(T).any()

# 4. UMAP muestra separabilidad
# Verificar visualmente que clusters existen
```

## 📦 Dependencias

```python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
umap-learn>=0.5.3
scikit-learn>=1.0.0
kaggle>=1.5.12
```

## 🚀 Cómo Ejecutar

### Paso 1: Preparar datos en Kaggle

1. Subir archivos .dat a Kaggle Dataset
2. Incluir Sample.dat con coordenadas
3. Añadir kaggle.json con credenciales

### Paso 2: Ejecutar Construction

```python
# En Google Colab o Kaggle Kernel
# Ejecutar todas las celdas de Construction.ipynb
# Output: Arrays en memoria (MS, T, Jex2, ...)
```

### Paso 3: Ejecutar DescriptionRescale

```python
# Ejecutar todas las celdas
# Visualizar UMAP plots
# Verificar distribuciones
# Upload a Kaggle
```

### Paso 4: Verificar Output

```python
# Descargar dataset
!kaggle datasets download -d deathperminut/material-spinners-data

# Cargar y verificar
data = np.load('data.npz')
print(data.files)  # Debe mostrar: ['MS', 'T', 'Jex2', ...]
```

## 🐛 Troubleshooting

**Problema:** "ERRORFILE: States1_02.dat"
- **Causa:** Archivo corrupto o formato incorrecto
- **Solución:** Filtrar archivo problemático antes del loop

**Problema:** UMAP no muestra clusters
- **Causa:** Parámetros del UMAP o datos muy ruidosos
- **Solución:** Ajustar `n_neighbors`, `min_dist`, o verificar calidad de simulaciones

**Problema:** Memory Error en Construction
- **Causa:** Demasiadas imágenes para RAM disponible
- **Solución:** Procesar en batches y guardar parciales

## 📊 Estadísticas de Datasets

### DatabaseJex2T

| Parámetro | Min | Max | Mean | Std |
|-----------|-----|-----|------|-----|
| **Temperatura (T)** | Variable | Variable | Variable | Variable |
| **Jex2** | Variable | Variable | Variable | Variable |
| **Imágenes (MS)** | -1.0 | +1.0 | ~0.0 | Variable |

### DatabaseKDMT

| Parámetro | Min | Max | Mean | Std |
|-----------|-----|-----|------|-----|
| **Temperatura (T)** | Variable | Variable | Variable | Variable |
| **KDM** | Variable | Variable | Variable | Variable |
| **Imágenes (MS)** | -1.0 | +1.0 | ~0.0 | Variable |

## 🔗 Siguiente Paso

Una vez completado el preprocessing:
→ Ir a `../Models/` para entrenamiento de redes neuronales

---

**Nota:** Los valores específicos de Min/Max/Mean dependen de los parámetros de simulación elegidos. Consultar notebooks para estadísticas exactas.
