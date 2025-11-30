# Referencias Bibliográficas

Sistema de gestión de referencias usando Zotero + Better BibTeX.

## Archivo Principal

- **biblioteca.bib**: Archivo BibTeX exportado automáticamente desde Zotero
  - Contiene todas las referencias bibliográficas
  - Se actualiza automáticamente cuando añades papers en Zotero

## Configuración de Zotero

### 1. Instalar Better BibTeX
- Descargar desde: https://retorque.re/zotero-better-bibtex/
- En Zotero: Tools → Add-ons → Install Add-on from File

### 2. Exportar Biblioteca
1. Click derecho en "My Library"
2. Export Library...
3. Format: **Better BibTeX**
4. ✓ Keep updated
5. Guardar como: `referencias/biblioteca.bib`

### 3. Uso en LaTeX

```latex
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[backend=bibtex]{biblatex}

\addbibresource{../referencias/biblioteca.bib}

\begin{document}

Según \cite{author2024}, ...

\printbibliography

\end{document}
```

## Organización de PDFs

Los PDFs se guardan en **Google Drive**, NO en este repositorio.

Estructura recomendada en Drive:
```
Google Drive/Doctorado/
└── Referencias-PDFs/
    ├── Machine Learning/
    ├── Statistics/
    └── Domain Specific/
```

## Papers Clave

Mantén aquí un índice de papers fundamentales para tu investigación:

### Metodología
- [ ] Autor et al. (2024) - Título del paper

### Estado del Arte
- [ ] Autor et al. (2023) - Título del paper

### Fundamentos Teóricos
- [ ] Autor et al. (2022) - Título del paper

---

**Nota**: El archivo `biblioteca.bib` se actualiza automáticamente. No lo edites manualmente.
