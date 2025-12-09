#!/usr/bin/env python3
"""
Script mejorado para convertir SVG a PDF sin recortar
Usa cairosvg que respeta el viewBox completo del SVG
"""
import os
import sys
from pathlib import Path

try:
    import cairosvg
except ImportError:
    print("Error: cairosvg no está instalado")
    print("Instalando cairosvg...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg"])
    import cairosvg

# Lista de SVG que necesitamos convertir
svg_files = [
    'pipeline_vertical_branch.svg',
    'imagenes_grid.svg',
    'imagenes_grid2.svg',
    'distribuciones_parametrosKDM.svg',
    'distribuciones_parametrosTJex2.svg',
    'comparacion_modelos_separados_KDM.svg',
    'comparacion_modelos_separados_JEX2.svg',
    'gradcam_1.svg',
    'mean_contribution_gradcam2.svg'
]

figures_dir = Path('/Users/juansebastianmendezrondon/Projects/Doctorado/papers/EstimationPaper1/Figures')

converted = 0
failed = 0

for svg_file in svg_files:
    svg_path = figures_dir / svg_file
    pdf_path = figures_dir / svg_file.replace('.svg', '.pdf')

    if svg_path.exists():
        try:
            print(f"Converting {svg_file}...")

            # Convertir SVG a PDF usando cairosvg
            # Esta librería respeta el viewBox y no recorta
            cairosvg.svg2pdf(
                url=str(svg_path),
                write_to=str(pdf_path),
                output_width=None,  # Respetar tamaño original
                output_height=None  # Respetar tamaño original
            )

            print(f"  ✓ Created {pdf_path.name}")
            converted += 1

        except Exception as e:
            print(f"  ✗ Failed to convert {svg_file}: {e}")
            failed += 1
    else:
        print(f"  ✗ {svg_file} not found")
        failed += 1

print(f"\nDone! Converted: {converted}, Failed: {failed}")
