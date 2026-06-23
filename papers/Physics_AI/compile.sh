#!/bin/bash
# Compilación del paper Physics_AI (VAM_SpinesDL.tex)
#
# Este proyecto viene de Overleaf y usa rutas mixtas:
#   - unas relativas a papers/ (ej: Physics_AI/figures/..., Physics_AI/literature)
#   - otras relativas a papers/Physics_AI/ (ej: Assets/..., figures/...)
# Por eso se compila DESDE aquí con TEXINPUTS/BIBINPUTS apuntando también al
# directorio padre (papers/), así resuelven ambos estilos de ruta.
#
# Uso: ./compile.sh           (compila VAM_SpinesDL.tex)
#      ./compile.sh -c         (compila y limpia auxiliares)

set -e
export PATH="/Library/TeX/texbin:$PATH"
export TEXINPUTS=".:..:"
export BIBINPUTS=".:..:"
export BSTINPUTS=".:..:"

TEXFILE="VAM_SpinesDL"
CLEAN=false
[ "$1" = "-c" ] || [ "$1" = "--clean" ] && CLEAN=true

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${YELLOW}▸ Pase 1/4: pdflatex (detecta citas)...${NC}"
pdflatex -interaction=nonstopmode -shell-escape "${TEXFILE}.tex" > compile_step1.log 2>&1

echo -e "${YELLOW}▸ Pase 2/4: bibtex (bibliografía)...${NC}"
bibtex "${TEXFILE}" > bibtex.log 2>&1 || true

echo -e "${YELLOW}▸ Pase 3/4: pdflatex (incorpora bibliografía)...${NC}"
pdflatex -interaction=nonstopmode -shell-escape "${TEXFILE}.tex" > compile_step2.log 2>&1

echo -e "${YELLOW}▸ Pase 4/4: pdflatex (resuelve referencias)...${NC}"
pdflatex -interaction=nonstopmode -shell-escape "${TEXFILE}.tex" > compile_step3.log 2>&1

# Reportar errores fatales si los hay
if grep -qE "^! " compile_step3.log; then
    echo -e "${RED}✗ Errores fatales encontrados:${NC}"
    grep -nE "^! " compile_step3.log | head -10
    exit 1
fi

UNDEF=$(grep -ciE "citation.*undefined" compile_step3.log || true)
[ "$UNDEF" -gt 0 ] && echo -e "${YELLOW}⚠ ${UNDEF} cita(s) sin definir${NC}"

if [ -f "${TEXFILE}.pdf" ]; then
    SIZE=$(ls -lh "${TEXFILE}.pdf" | awk '{print $5}')
    echo -e "${GREEN}✓ PDF generado: ${TEXFILE}.pdf (${SIZE})${NC}"
fi

if [ "$CLEAN" = true ]; then
    rm -f ${TEXFILE}.aux ${TEXFILE}.log ${TEXFILE}.out ${TEXFILE}.toc ${TEXFILE}.blg ${TEXFILE}.bbl
    rm -f compile_step*.log bibtex.log
    echo -e "${GREEN}✓ Auxiliares limpiados${NC}"
fi
