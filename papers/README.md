# Papers - Escritura Académica en LaTeX

## 📋 Descripción

Esta carpeta contiene todos los papers académicos en formato LaTeX. Puedes trabajar localmente con VS Code (con preview en tiempo real) o sincronizar con Overleaf para colaboración.

## 🗂️ Estructura

```
papers/
├── README.md               # Esta guía
├── EstimationPaper1/       # Paper principal
│   ├── main.tex           # Documento principal
│   ├── references.bib     # Bibliografía
│   ├── figures/           # Figuras e imágenes
│   └── sections/          # Secciones separadas (opcional)
│
└── PreliminaryDraft/       # Borrador preliminar
    └── ...
```

## 🚀 Cómo Trabajar con LaTeX en VS Code

### Opción 1: Compilación Automática (Recomendada)

1. **Abrir VS Code en la carpeta del proyecto:**
   ```bash
   cd ~/Projects/Doctorado
   code .
   ```

2. **Abrir el archivo main.tex:**
   - Navega a `papers/EstimationPaper1/main.tex`

3. **Compilar y ver PDF:**
   - Guarda el archivo (Cmd + S)
   - LaTeX Workshop compila automáticamente
   - Ver PDF: Click en el ícono "View LaTeX PDF" (arriba derecha)
   - O usa: `Cmd + Shift + P` → "LaTeX Workshop: View LaTeX PDF"

4. **Split View (Código | PDF):**
   - Arrastra la pestaña del PDF a la derecha
   - Ahora ves código y PDF lado a lado
   - Al guardar, el PDF se actualiza automáticamente

### Opción 2: Comandos Manuales

```bash
cd papers/EstimationPaper1

# Compilar
pdflatex main.tex

# Si tienes bibliografía
bibtex main
pdflatex main.tex
pdflatex main.tex  # Sí, dos veces para resolver referencias

# Ver PDF
open main.pdf
```

## ⚙️ Configuración de VS Code

### Extensiones Instaladas

✅ **LaTeX Workshop** - Ya instalado, proporciona:
- Compilación automática
- Preview de PDF
- Autocompletado de comandos
- Detección de errores
- SyncTeX (click en PDF → código)

### Atajos de Teclado Útiles

| Atajo | Acción |
|-------|--------|
| `Cmd + S` | Guardar y compilar |
| `Cmd + Shift + P` | Command Palette |
| `Cmd + Option + B` | Build LaTeX |
| `Cmd + Option + V` | Ver PDF |
| `Cmd + Option + J` | SyncTeX (PDF → código) |

### Configuración Personalizada (.vscode/settings.json)

Ya está configurado con:
- Compilación automática al guardar
- Limpieza de archivos auxiliares
- Viewer integrado en VS Code

## 📝 Estructura de un Paper

### Archivo Principal (main.tex)

```latex
\documentclass[12pt,a4paper]{article}

% Paquetes
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}

\title{Título del Paper}
\author{Tu Nombre}

\begin{document}
\maketitle

\section{Introducción}
Contenido...

\bibliography{references}
\end{document}
```

### Archivo de Bibliografía (references.bib)

```bibtex
@article{autor2024,
    author = {Autor, A.},
    title = {Título del Paper},
    journal = {Journal Name},
    year = {2024},
    volume = {10},
    pages = {1-10}
}
```

### Citar en el Texto

```latex
Según \cite{autor2024}, los resultados muestran...

Múltiples citas \cite{autor2024, otro2023}.
```

## 🖼️ Figuras

### Añadir Figuras

1. **Guardar figura en `figures/`:**
   ```
   figures/
   ├── scatter_plot.pdf
   ├── umap_visualization.png
   └── gradcam_heatmap.pdf
   ```

2. **Incluir en LaTeX:**
   ```latex
   \begin{figure}[H]
   \centering
   \includegraphics[width=0.7\textwidth]{figures/scatter_plot.pdf}
   \caption{Descripción de la figura}
   \label{fig:scatter}
   \end{figure}
   ```

3. **Referenciar:**
   ```latex
   Como se muestra en la Figura \ref{fig:scatter}...
   ```

### Formatos Recomendados

- **Gráficos:** PDF o SVG (vectorial, mejor calidad)
- **Fotos:** PNG o JPEG
- **Evitar:** BMP (muy pesado)

## 📚 Integración con Zotero

### Setup Automático

Cuando configures Zotero + Better BibTeX:

1. **Exportar biblioteca:**
   - Click derecho en biblioteca → Export
   - Format: Better BibTeX
   - ✓ Keep updated
   - Guardar en: `referencias/biblioteca.bib`

2. **Usar en LaTeX:**
   ```latex
   \bibliography{../referencias/biblioteca}
   ```

### Alternativa: Copiar Manualmente

```bibtex
# En references.bib, pega entradas de Zotero
@article{key,
    author = {...},
    ...
}
```

## 🔄 Sincronización con Overleaf

### Opción A: GitHub Sync (Premium/Institucional)

Si tienes Overleaf premium:

1. En Overleaf: Menu → GitHub
2. Link repositorio
3. Trabaja en VS Code localmente
4. Push a GitHub cuando quieras actualizar Overleaf

### Opción B: Manual (Gratuita)

**Local → Overleaf:**
1. Comprimir carpeta:
   ```bash
   cd papers
   zip -r EstimationPaper1.zip EstimationPaper1/
   ```
2. En Overleaf: New Project → Upload Project → Subir ZIP

**Overleaf → Local:**
1. En Overleaf: Menu → Download → Source
2. Descomprimir en `papers/`
3. Commit a Git

## 🐛 Solución de Problemas

### Error: "pdflatex not found"

**Problema:** MacTeX no instalado o no en PATH

**Solución:**
```bash
# Reiniciar terminal
eval "$(/usr/libexec/path_helper)"

# Verificar
which pdflatex
```

### Error: "Undefined control sequence"

**Problema:** Comando LaTeX incorrecto o paquete faltante

**Solución:**
- Verificar sintaxis del comando
- Añadir `\usepackage{...}` necesario

### PDF no se actualiza

**Problema:** LaTeX Workshop no detecta cambios

**Solución:**
- Cmd + Shift + P → "LaTeX Workshop: Build LaTeX project"
- O borrar archivos auxiliares: `rm *.aux *.log`

### Bibliografía no aparece

**Problema:** BibTeX no ejecutado

**Solución:**
```bash
pdflatex main.tex
bibtex main        # ← Importante
pdflatex main.tex
pdflatex main.tex
```

## 📊 Workflow Recomendado

### 1. Escritura Diaria

```bash
# Mañana
cd ~/Projects/Doctorado
code .

# Abrir main.tex
# Escribir, guardar (compila automático)
# Ver PDF en split view
```

### 2. Añadir Figuras

```bash
# Desde Results notebooks, exportar SVG
# Copiar a papers/EstimationPaper1/figures/

# En LaTeX:
\includegraphics{figures/mi_figura.pdf}
```

### 3. Actualizar Bibliografía

```bash
# Desde Zotero, exportar a referencias/biblioteca.bib
# O editar references.bib manualmente
```

### 4. Guardar en Git

```bash
git add papers/
git commit -m "Paper: añadida sección de resultados"
git push
```

## 📋 Checklist Antes de Enviar

- [ ] Compilación sin errores
- [ ] Todas las figuras incluidas y referenciadas
- [ ] Bibliografía completa y citada
- [ ] Formato según journal template
- [ ] Spell check (VS Code: Code Spell Checker extension)
- [ ] Números de sección correctos
- [ ] Abstract < 250 palabras
- [ ] Figuras en alta resolución

## 🎨 Templates Comunes

### Paper de Revista

Ya incluido en `EstimationPaper1/main.tex`

### Paper de Conferencia (IEEE)

```latex
\documentclass[conference]{IEEEtran}
% ...
```

### Thesis Chapter

```latex
\documentclass[12pt]{report}
\chapter{Capítulo 1}
% ...
```

## 🔗 Recursos

- [Overleaf Learn](https://www.overleaf.com/learn) - Tutorial completo de LaTeX
- [Detexify](http://detexify.kirelabs.org/classify.html) - Encuentra símbolos LaTeX
- [Tables Generator](https://www.tablesgenerator.com/) - Genera tablas LaTeX
- [TikZ](https://tikz.dev/) - Diagramas y figuras en LaTeX

## 🆘 Ayuda Rápida

### Comandos Matemáticos

```latex
% Inline
$E = mc^2$

% Display
\begin{equation}
E = mc^2
\label{eq:einstein}
\end{equation}

% Referenciar
La Ecuación \ref{eq:einstein} muestra...
```

### Tablas

```latex
\begin{table}[H]
\centering
\caption{Resultados}
\begin{tabular}{lcc}
\toprule
Modelo & R² & MAPE \\
\midrule
DenseNet & 0.9753 & 18.64\% \\
ResNet & 0.94 & 22\% \\
\bottomrule
\end{tabular}
\end{table}
```

### Listas

```latex
% Enumerada
\begin{enumerate}
\item Primero
\item Segundo
\end{enumerate}

% Bullets
\begin{itemize}
\item Punto 1
\item Punto 2
\end{itemize}
```

---

**Nota:** La primera compilación puede tardar más (genera archivos auxiliares). Compilaciones subsiguientes son rápidas (~1-2 segundos).
