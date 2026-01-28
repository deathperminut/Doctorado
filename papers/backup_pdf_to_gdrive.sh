#!/bin/bash

# Script para backup automático de PDFs de papers a Google Drive
# Uso: ./backup_pdf_to_gdrive.sh [paper_name]
# Ejemplo: ./backup_pdf_to_gdrive.sh PreliminaryDraft
#          ./backup_pdf_to_gdrive.sh EstimationPaper1
#          ./backup_pdf_to_gdrive.sh OptimizationData

# Configuración
REMOTE_NAME="gdrive"  # Nombre del remote de rclone (cambiar si usaste otro nombre)
BASE_REMOTE_PATH="DoctoradoMacConfig/Papers"  # Carpeta base en Google Drive
BASE_LOCAL_PATH="/Users/juansebastianmendezrondon/Projects/Doctorado/papers"

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para mostrar uso
show_usage() {
    echo -e "${BLUE}=== Backup de PDF a Google Drive ===${NC}"
    echo ""
    echo "Uso: $0 [paper_name]"
    echo ""
    echo "Papers disponibles:"
    echo "  - PreliminaryDraft"
    echo "  - EstimationPaper1"
    echo "  - OptimizationData"
    echo ""
    echo "Ejemplo: $0 PreliminaryDraft"
    echo ""
    exit 1
}

# Verificar parámetros
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: Debes especificar el nombre del paper${NC}"
    echo ""
    show_usage
fi

PAPER_NAME="$1"
PAPER_PATH="${BASE_LOCAL_PATH}/${PAPER_NAME}"
PDF_FILE="${PAPER_PATH}/main.pdf"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATE=$(date +"%Y-%m-%d %H:%M:%S")

echo -e "${BLUE}=== Backup de PDF a Google Drive ===${NC}"
echo -e "Paper: ${GREEN}${PAPER_NAME}${NC}"
echo -e "Fecha: ${DATE}"
echo ""

# Verificar que la carpeta del paper existe
if [ ! -d "$PAPER_PATH" ]; then
    echo -e "${RED}Error: Carpeta '$PAPER_PATH' no encontrada${NC}"
    echo ""
    echo "Carpetas disponibles en papers/:"
    ls -d "${BASE_LOCAL_PATH}"/*/ 2>/dev/null | xargs -n 1 basename
    echo ""
    exit 1
fi

# Verificar que el PDF existe
if [ ! -f "$PDF_FILE" ]; then
    echo -e "${RED}Error: $PDF_FILE no encontrado${NC}"
    echo ""
    echo -e "${YELLOW}Tip: Compila el documento LaTeX primero con:${NC}"
    echo "  cd $PAPER_PATH"
    echo "  pdflatex main.tex"
    echo ""
    exit 1
fi

# Obtener tamaño del archivo
FILE_SIZE=$(ls -lh "$PDF_FILE" | awk '{print $5}')
NUM_PAGES=$(pdfinfo "$PDF_FILE" 2>/dev/null | grep "Pages:" | awk '{print $2}')
echo -e "Archivo: ${GREEN}main.pdf${NC}"
echo -e "Tamaño: ${FILE_SIZE}"
if [ ! -z "$NUM_PAGES" ]; then
    echo -e "Páginas: ${NUM_PAGES}"
fi
echo ""

# Verificar que rclone está instalado
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}Error: rclone no está instalado${NC}"
    echo ""
    echo -e "${YELLOW}Instala rclone con:${NC}"
    echo "  brew install rclone"
    echo ""
    echo "Luego configura Google Drive con:"
    echo "  rclone config"
    echo ""
    exit 1
fi

# Verificar que el remote está configurado
if ! rclone listremotes | grep -q "^${REMOTE_NAME}:$"; then
    echo -e "${RED}Error: Remote '${REMOTE_NAME}' no configurado${NC}"
    echo ""
    echo -e "${YELLOW}Configura Google Drive con:${NC}"
    echo "  rclone config"
    echo ""
    echo "Pasos:"
    echo "  1. Nombre del remote: ${REMOTE_NAME}"
    echo "  2. Storage type: drive (Google Drive)"
    echo "  3. Sigue las instrucciones para autorizar"
    echo ""
    exit 1
fi

# Crear carpeta remota específica para este paper
REMOTE_PATH="${BASE_REMOTE_PATH}/${PAPER_NAME}"
echo -e "${BLUE}Preparando carpeta remota...${NC}"
rclone mkdir "${REMOTE_NAME}:${REMOTE_PATH}" 2>/dev/null
echo -e "${GREEN}✓ Carpeta: ${REMOTE_NAME}:${REMOTE_PATH}/${NC}"
echo ""

# Backup con timestamp (guardar historial)
echo -e "${BLUE}[1/2] Subiendo versión timestamped...${NC}"
BACKUP_NAME="main_${TIMESTAMP}.pdf"
TEMP_BACKUP="/tmp/${BACKUP_NAME}"
cp "$PDF_FILE" "$TEMP_BACKUP"

if rclone copy "$TEMP_BACKUP" "${REMOTE_NAME}:${REMOTE_PATH}/" --progress; then
    echo -e "${GREEN}✓ Guardado como: ${BACKUP_NAME}${NC}"
    rm "$TEMP_BACKUP"
else
    echo -e "${RED}✗ Error al subir archivo timestamped${NC}"
    rm "$TEMP_BACKUP"
    exit 1
fi
echo ""

# Backup como "main.pdf" (versión actual siempre disponible)
echo -e "${BLUE}[2/2] Actualizando versión actual...${NC}"
if rclone copy "$PDF_FILE" "${REMOTE_NAME}:${REMOTE_PATH}/" --progress; then
    echo -e "${GREEN}✓ Actualizado: main.pdf (versión actual)${NC}"
else
    echo -e "${RED}✗ Error al actualizar versión actual${NC}"
    exit 1
fi
echo ""

# Mostrar resumen
echo -e "${GREEN}=== Backup Completado Exitosamente ===${NC}"
echo ""
echo -e "Paper: ${GREEN}${PAPER_NAME}${NC}"
echo -e "Ubicación remota: ${BLUE}${REMOTE_NAME}:${REMOTE_PATH}/${NC}"
echo ""
echo -e "Archivos guardados:"
echo -e "  ✓ ${BACKUP_NAME} (versión con timestamp)"
echo -e "  ✓ main.pdf (versión actual)"
echo ""

# Mostrar últimos backups de este paper
echo -e "${BLUE}Últimos 5 backups de ${PAPER_NAME}:${NC}"
rclone ls "${REMOTE_NAME}:${REMOTE_PATH}/" 2>/dev/null | grep "main_" | tail -5 | while read size file; do
    echo -e "  • ${file} (${size} bytes)"
done

echo ""
echo -e "${GREEN}✓ Proceso completado${NC}"
echo ""
echo -e "${YELLOW}Tip:${NC} Para ver tus archivos en Google Drive:"
echo "  rclone ls ${REMOTE_NAME}:${REMOTE_PATH}/"
