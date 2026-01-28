#!/bin/bash

# Script para listar backups existentes en Google Drive
# Uso: ./list_backups.sh [paper_name]

REMOTE_NAME="gdrive"
BASE_REMOTE_PATH="DoctoradoMacConfig/Papers"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ $# -eq 0 ]; then
    echo -e "${BLUE}=== Backups en Google Drive ===${NC}"
    echo ""

    # Listar todos los papers
    for paper in PreliminaryDraft EstimationPaper1 OptimizationData; do
        REMOTE_PATH="${BASE_REMOTE_PATH}/${paper}"
        echo -e "${GREEN}${paper}:${NC}"

        COUNT=$(rclone ls "${REMOTE_NAME}:${REMOTE_PATH}/" 2>/dev/null | wc -l)
        if [ $COUNT -gt 0 ]; then
            rclone ls "${REMOTE_NAME}:${REMOTE_PATH}/" 2>/dev/null | tail -5 | while read size file; do
                echo -e "  • ${file}"
            done
        else
            echo -e "  ${YELLOW}(Sin backups)${NC}"
        fi
        echo ""
    done
else
    PAPER_NAME="$1"
    REMOTE_PATH="${BASE_REMOTE_PATH}/${PAPER_NAME}"

    echo -e "${BLUE}=== Backups de ${PAPER_NAME} ===${NC}"
    echo ""

    rclone ls "${REMOTE_NAME}:${REMOTE_PATH}/" 2>/dev/null | while read size file; do
        # Convertir tamaño a formato legible
        size_mb=$(echo "scale=2; $size / 1048576" | bc)
        echo -e "  • ${file} (${size_mb} MB)"
    done
fi
