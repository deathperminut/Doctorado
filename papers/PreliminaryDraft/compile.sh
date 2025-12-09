#!/bin/bash

# Script de compilación LaTeX con BibTeX
# Uso: ./compile.sh [nombre_archivo.tex] [opciones]
# Si no se especifica archivo, busca main.tex o el único .tex en el directorio

set -e  # Detener en caso de error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Agregar MacTeX al PATH
export PATH="/Library/TeX/texbin:$PATH"

# Función para mostrar uso
show_usage() {
    echo "Uso: $0 [ARCHIVO] [OPCIONES]"
    echo ""
    echo "ARCHIVO: nombre del archivo .tex (sin extensión). Por defecto: main"
    echo ""
    echo "OPCIONES:"
    echo "  -c, --clean    Limpiar archivos auxiliares después de compilar"
    echo "  -q, --quiet    Modo silencioso (solo errores)"
    echo "  -h, --help     Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo "  $0              # Compila main.tex"
    echo "  $0 article      # Compila article.tex"
    echo "  $0 main -c      # Compila y limpia archivos auxiliares"
}

# Variables por defecto
CLEAN=false
QUIET=false
TEXFILE=""

# Parsear argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo -e "${RED}Error: Opción desconocida: $1${NC}"
            show_usage
            exit 1
            ;;
        *)
            if [ -z "$TEXFILE" ]; then
                TEXFILE="$1"
            else
                echo -e "${RED}Error: Solo se puede especificar un archivo${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Si no se especificó archivo, buscar main.tex o el único .tex
if [ -z "$TEXFILE" ]; then
    if [ -f "main.tex" ]; then
        TEXFILE="main"
    else
        # Contar archivos .tex en el directorio
        TEX_COUNT=$(ls -1 *.tex 2>/dev/null | wc -l)
        if [ "$TEX_COUNT" -eq 1 ]; then
            TEXFILE=$(ls *.tex | sed 's/\.tex$//')
        elif [ "$TEX_COUNT" -eq 0 ]; then
            echo -e "${RED}Error: No se encontró ningún archivo .tex en el directorio${NC}"
            exit 1
        else
            echo -e "${RED}Error: Múltiples archivos .tex encontrados. Especifica cuál compilar.${NC}"
            ls *.tex
            exit 1
        fi
    fi
fi

# Remover extensión .tex si se proporcionó
TEXFILE="${TEXFILE%.tex}"

# Verificar que el archivo existe
if [ ! -f "${TEXFILE}.tex" ]; then
    echo -e "${RED}Error: No se encontró el archivo ${TEXFILE}.tex${NC}"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Compilación LaTeX: ${TEXFILE}.tex${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo ""

# Función para ejecutar comandos
run_cmd() {
    local cmd="$1"
    local description="$2"
    local log_file="$3"

    if [ "$QUIET" = true ]; then
        if $cmd > "$log_file" 2>&1; then
            echo -e "${GREEN}✓${NC} $description"
            return 0
        else
            echo -e "${RED}✗${NC} $description"
            echo -e "${RED}Ver errores en: $log_file${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}▸${NC} $description..."
        if $cmd > "$log_file" 2>&1; then
            echo -e "${GREEN}✓${NC} $description completado"
            return 0
        else
            echo -e "${RED}✗${NC} Error en $description"
            echo -e "${RED}Ver errores en: $log_file${NC}"
            tail -20 "$log_file"
            return 1
        fi
    fi
}

# Paso 1: Primera compilación pdflatex
run_cmd "pdflatex -interaction=nonstopmode ${TEXFILE}.tex" \
        "Paso 1/4: Primera compilación (detectando citas)" \
        "compile_step1.log"

# Paso 2: BibTeX (solo si existe archivo .bib)
if ls *.bib >/dev/null 2>&1; then
    run_cmd "bibtex ${TEXFILE}" \
            "Paso 2/4: Procesando bibliografía (BibTeX)" \
            "bibtex.log"
else
    echo -e "${YELLOW}⚠${NC}  Paso 2/4: No se encontró archivo .bib, saltando BibTeX"
fi

# Paso 3: Segunda compilación pdflatex
run_cmd "pdflatex -interaction=nonstopmode ${TEXFILE}.tex" \
        "Paso 3/4: Segunda compilación (incorporando bibliografía)" \
        "compile_step2.log"

# Paso 4: Tercera compilación pdflatex
run_cmd "pdflatex -interaction=nonstopmode ${TEXFILE}.tex" \
        "Paso 4/4: Tercera compilación (resolviendo referencias)" \
        "compile_step3.log"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ Compilación completada exitosamente${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"

# Mostrar información del PDF generado
if [ -f "${TEXFILE}.pdf" ]; then
    PDF_SIZE=$(ls -lh "${TEXFILE}.pdf" | awk '{print $5}')
    echo -e "${BLUE}📄 PDF generado:${NC} ${TEXFILE}.pdf (${PDF_SIZE})"
fi

# Verificar si hay citas sin definir
UNDEFINED_CITES=$(grep -i "citation.*undefined" "${TEXFILE}.log" 2>/dev/null | wc -l || echo 0)
if [ "$UNDEFINED_CITES" -gt 0 ]; then
    echo -e "${YELLOW}⚠  Advertencia: ${UNDEFINED_CITES} cita(s) sin definir${NC}"
    grep -i "citation.*undefined" "${TEXFILE}.log" | head -5
fi

# Limpiar archivos auxiliares si se solicitó
if [ "$CLEAN" = true ]; then
    echo ""
    echo -e "${YELLOW}▸${NC} Limpiando archivos auxiliares..."
    rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
    rm -f compile_step*.log bibtex.log
    echo -e "${GREEN}✓${NC} Archivos auxiliares eliminados"
fi

echo ""
