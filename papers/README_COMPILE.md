# Script de Compilación LaTeX

Script automatizado para compilar documentos LaTeX con bibliografía BibTeX.

## Instalación

Para usar el script en un nuevo artículo, copia `compile.sh` al directorio de tu proyecto:

```bash
cp ~/Projects/Doctorado/papers/compile.sh ./
chmod +x compile.sh
```

## Uso Básico

### Compilar main.tex (por defecto)
```bash
./compile.sh
```

### Compilar un archivo específico
```bash
./compile.sh article
./compile.sh my_paper.tex
```

### Compilar y limpiar archivos auxiliares
```bash
./compile.sh -c
./compile.sh main -c
```

### Modo silencioso (solo errores)
```bash
./compile.sh -q
```

## Opciones

| Opción | Descripción |
|--------|-------------|
| `-c, --clean` | Elimina archivos auxiliares (.aux, .log, etc.) después de compilar |
| `-q, --quiet` | Modo silencioso, solo muestra errores |
| `-h, --help` | Muestra la ayuda |

## Ejemplos

```bash
# Compilación normal con output detallado
./compile.sh

# Compilación silenciosa del archivo "draft.tex"
./compile.sh draft -q

# Compilar y limpiar todo
./compile.sh main --clean

# Ver ayuda
./compile.sh --help
```

## Qué hace el script

El script ejecuta automáticamente el ciclo completo de compilación LaTeX:

1. **Primera compilación** (`pdflatex`) - Detecta citas y referencias
2. **BibTeX** - Procesa la bibliografía desde archivos `.bib`
3. **Segunda compilación** (`pdflatex`) - Incorpora la bibliografía
4. **Tercera compilación** (`pdflatex`) - Resuelve referencias cruzadas

## Solución de Problemas

### El script no encuentra pdflatex
El script ya incluye la ruta de MacTeX: `/Library/TeX/texbin`

Si tienes otra instalación de TeX, edita la línea:
```bash
export PATH="/Library/TeX/texbin:$PATH"
```

### Errores de compilación
Los logs se guardan en:
- `compile_step1.log` - Primera compilación
- `bibtex.log` - Procesamiento BibTeX
- `compile_step2.log` - Segunda compilación
- `compile_step3.log` - Tercera compilación

### Citas sin definir
Si ves advertencias de "citas sin definir":
1. Verifica que las referencias existan en tu archivo `.bib`
2. Asegúrate de que los nombres de las citas coincidan exactamente

## Archivos Generados

### Archivos importantes
- `main.pdf` - Tu documento compilado

### Archivos auxiliares (se pueden eliminar con `-c`)
- `*.aux` - Información auxiliar
- `*.bbl` - Bibliografía procesada
- `*.blg` - Log de BibTeX
- `*.log` - Logs de compilación
- `*.out` - Enlaces de hyperref
- `*.toc` - Tabla de contenidos
- Y otros archivos temporales

---

**Creado**: Diciembre 2025
**Autor**: Juan Sebastián Méndez Rondón
**Proyecto**: PhD Thesis - Magnetic Domain Characterization
