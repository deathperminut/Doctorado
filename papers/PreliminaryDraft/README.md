# Preliminary Draft - Borrador Preliminar

## 📋 Descripción

Este es un borrador de trabajo para organizar ideas, resultados preliminares y notas durante el desarrollo del proyecto.

**Propósito:**
- Documentar avances incrementales
- Probar ideas y estructuras
- Espacio libre para experimentar con LaTeX
- Versión "sucia" antes del paper final

## 🎯 Diferencias con EstimationPaper1

| Aspecto | PreliminaryDraft | EstimationPaper1 |
|---------|------------------|------------------|
| Propósito | Borrador de trabajo | Paper final |
| Estructura | Flexible | Formal |
| Completitud | Parcial, en construcción | Completo |
| Audiencia | Tú mismo, advisor | Journal, revisores |

## 📝 Cómo Usar Este Borrador

### 1. Para Notas Rápidas

Usa este documento para:
- Pegar resultados temporales
- Probar estructuras de secciones
- Experimentar con ecuaciones
- Guardar snippets de código LaTeX

### 2. Para Revisiones con Advisor

Compila el PDF y compártelo para:
- Mostrar avances
- Recibir feedback temprano
- Iterar sobre estructura
- Validar dirección

### 3. Como Base para Paper Final

Cuando esté listo:
- Copia secciones maduras a `EstimationPaper1/`
- Refina el contenido
- Añade rigor formal

## 🛠️ Compilar

### En VS Code:
1. Abre `main.tex`
2. Guarda (Cmd + S)
3. PDF se genera automáticamente

### Manual:
```bash
cd papers/PreliminaryDraft
pdflatex main.tex
open main.pdf
```

## 📂 Estructura

```
PreliminaryDraft/
├── main.tex           # Borrador principal
├── references.bib     # Referencias básicas
├── figures/           # Figuras temporales
└── notes/             # Notas sueltas
```

## ✏️ Secciones Incluidas

- ✅ Abstract preliminar
- ✅ Introducción y objetivos
- ✅ Metodología (esqueleto)
- ✅ Resultados (placeholder)
- ✅ Próximos pasos
- ✅ Notas y observaciones
- ✅ Ideas y dudas

## 🎨 Tips para Trabajar con Borradores

1. **No te preocupes por perfección:** Este es tu espacio de trabajo
2. **Usa comentarios:** `% TODO: completar esta sección`
3. **Deja placeholders:** `\textit{[En construcción]}`
4. **Versiona frecuentemente:** Commit a Git regularmente
5. **Itera rápido:** Prueba, compila, ajusta

## 📊 Añadir Contenido

### Resultados Rápidos

```latex
\section{Experimento 2025-01-15}

Probé DenseNet con lr=1e-4:

\begin{itemize}
\item R²: 0.95
\item MAPE: 20\%
\item Observaciones: Converge rápido pero overfit
\end{itemize}
```

### Figuras Temporales

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{figures/test_plot.png}
\caption{Prueba de visualización - eliminar después}
\end{figure}
```

### TODOs

```latex
% TODO: Añadir tabla comparativa de modelos
% TODO: Verificar estos números con notebook
% FIXME: Esta ecuación tiene error de signo
```

## 🔄 Workflow Sugerido

```
1. Experimento en notebook
   ↓
2. Resultados iniciales → PreliminaryDraft
   ↓
3. Feedback del advisor
   ↓
4. Iterar
   ↓
5. Cuando esté maduro → EstimationPaper1
```

## 📝 Comandos LaTeX Útiles para Borradores

```latex
% Resaltar texto
\textbf{IMPORTANTE: revisar esto}

% Notas temporales
\textit{[Pendiente: añadir análisis]}

% Espacio para desarrollar
\vspace{2cm}
% [Espacio para tabla]

% Comentarios largos
\begin{comment}
Esta sección fue descartada porque...
Mantener por si acaso.
\end{comment}
```

---

**Recuerda:** Este es tu playground de LaTeX. Experimenta sin miedo.
