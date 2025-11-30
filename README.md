# Doctorado - Sistema de Gestión de Investigación

Sistema centralizado para gestionar toda la investigación del doctorado: papers, código, notas, referencias bibliográficas y experimentos.

## Estructura del Proyecto

```
doctorado/
├── papers/              # Papers académicos en LaTeX
├── notebooks/           # Jupyter notebooks y análisis
├── notas/              # Obsidian vault - Sistema de notas
├── referencias/        # Bibliografía y gestión de citas
├── codigo/             # Scripts y utilidades reutilizables
└── datos/              # Scripts de descarga (datos en Kaggle/Drive)
```

## Herramientas del Sistema

### 1. Control de Versiones
- **GitHub**: Repositorio central con todo el trabajo versionado
- **Git**: Control de versiones local

### 2. Escritura Académica
- **LaTeX**: Formato para papers (compatible con Overleaf)
- **VS Code**: Editor local con preview de PDFs
- **Overleaf**: Opción para colaboración online

### 3. Gestión de Referencias
- **Zotero**: Gestor de bibliografía
- **Better BibTeX**: Plugin para exportar a `referencias/biblioteca.bib`

### 4. Notas y Conocimiento
- **Obsidian**: Sistema de notas en Markdown
- Vault ubicado en `notas/`
- Totalmente versionado con Git

### 5. Datos y Experimentos
- **Kaggle**: Almacenamiento de datasets
- **Google Colab**: Notebooks en la nube
- Scripts de descarga en `datos/`

### 6. Almacenamiento
- **GitHub**: Código, LaTeX, Markdown
- **Google Drive**: PDFs, datasets grandes, multimedia
- **Kaggle**: Datasets públicos

## Flujo de Trabajo

### Inicio del Día
```bash
cd ~/Projects/Doctorado
git pull origin main
```

### Durante el Día
- **Escribir papers**: Editar en `papers/` con VS Code o Overleaf
- **Tomar notas**: Usar Obsidian en `notas/`
- **Programar**: Crear notebooks en `notebooks/` o scripts en `codigo/`
- **Experimentar**: Google Colab conectado al repo

### Fin del Día
```bash
git add .
git commit -m "Descripción de avances"
git push origin main
```

## Google Colab - Setup

Para usar el repositorio en Colab:

```python
# 1. Clonar repositorio
!git clone https://github.com/deathperminut/Doctorado.git
%cd Doctorado

# 2. Descargar datos de Kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/  # Subir kaggle.json a Colab
!kaggle datasets download -d dataset-name

# 3. Trabajar...

# 4. Guardar cambios
!git config user.email "tu-email@example.com"
!git config user.name "Tu Nombre"
!git add notebooks/
!git commit -m "Resultados experimento"
!git push
```

## Estructura de Carpetas Detallada

### `/papers` - Papers Académicos
Cada paper en su propia carpeta:
```
papers/
├── 2024-paper-titulo/
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── bibliography.bib
```

### `/notebooks` - Análisis y Experimentos
Notebooks de Jupyter para exploración y análisis:
```
notebooks/
├── 01-exploratory-analysis.ipynb
├── 02-model-training.ipynb
└── 03-results-visualization.ipynb
```

### `/notas` - Sistema de Notas (Obsidian)
```
notas/
├── literatura/     # Resúmenes de papers
├── ideas/          # Ideas de investigación
├── meetings/       # Notas de reuniones
└── daily-notes/    # Diario de investigación
```

### `/referencias` - Bibliografía
```
referencias/
├── biblioteca.bib  # Exportado automáticamente de Zotero
└── README.md       # Índice de papers clave
```

### `/codigo` - Scripts Reutilizables
```
codigo/
├── preprocessing/  # Limpieza de datos
├── utils/          # Funciones auxiliares
└── analysis/       # Análisis estadístico
```

### `/datos` - Scripts de Descarga
```
datos/
├── download_from_kaggle.py
└── README.md  # Links a datasets en Kaggle/Drive
```

## Comandos Útiles

### Git
```bash
git status                  # Ver cambios
git add .                   # Añadir todos los cambios
git commit -m "mensaje"     # Crear commit
git push origin main        # Subir a GitHub
git pull origin main        # Descargar cambios
```

### LaTeX (local)
```bash
cd papers/mi-paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Integraciones

### Zotero → Git
1. Instalar **Better BibTeX** plugin
2. Click derecho en biblioteca → Export Library
3. Formato: Better BibTeX
4. Guardar en: `referencias/biblioteca.bib`
5. ✓ Keep updated

### Overleaf → Git
**Opción 1 (cuenta premium):**
- Menu → GitHub → Push/Pull

**Opción 2 (cuenta gratuita):**
- Menu → Download → Source
- Descomprimir en `papers/`
- `git add` y commit

### VS Code Extensions
Recomendadas:
- LaTeX Workshop
- GitHub Copilot (gratis para estudiantes)
- Foam (para Obsidian/Markdown)
- Python
- Jupyter

## Backups

Sistema de respaldo en 3 niveles:
1. **GitHub**: Código, papers, notas (automático con git push)
2. **Google Drive**: PDFs, datasets, multimedia
3. **Time Machine**: Backup completo local (configurar)

## Notas Importantes

- ❌ NO subas PDFs al repo (solo bibliografía .bib)
- ❌ NO subas datasets grandes (usa Kaggle/Drive)
- ❌ NO subas credenciales (kaggle.json, .env)
- ✅ Commit frecuentemente (mínimo una vez al día)
- ✅ Mensajes de commit descriptivos
- ✅ Mantén el .gitignore actualizado

## Recursos

- [Documentación LaTeX](https://www.overleaf.com/learn)
- [Obsidian Help](https://help.obsidian.md)
- [Zotero Documentation](https://www.zotero.org/support/)
- [GitHub Student Pack](https://education.github.com/pack)

---

**Última actualización**: 2025-11-22
