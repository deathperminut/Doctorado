# Datos

Esta carpeta contiene **scripts de descarga** de datos, NO los datos en sí.

## Ubicación de Datasets

Los datasets se almacenan en:
- **Kaggle**: Datasets públicos y competencias
- **Google Drive**: Datasets privados y procesados

## Scripts de Descarga

Ejemplo de script para descargar desde Kaggle:

```python
# download_from_kaggle.py
import os

# Asegurar que kaggle.json está configurado
# cp kaggle.json ~/.kaggle/

datasets = [
    "dataset-name-1",
    "dataset-name-2"
]

for dataset in datasets:
    os.system(f"kaggle datasets download -d {dataset}")
    os.system(f"unzip {dataset.split('/')[-1]}.zip")
```

## Datasets del Proyecto

| Dataset | Ubicación | Descripción | Tamaño |
|---------|-----------|-------------|--------|
| ejemplo | Kaggle: user/dataset | Descripción breve | 2GB |

## Uso en Google Colab

```python
# Descargar datos en Colab
!pip install kaggle
!mkdir -p ~/.kaggle
# Subir kaggle.json manualmente
!kaggle datasets download -d dataset-name
!unzip dataset-name.zip
```

---

⚠️ **IMPORTANTE**: Nunca subas archivos de datos grandes (.csv, .xlsx, .h5, etc.) a este repositorio. Usa Kaggle o Google Drive.
