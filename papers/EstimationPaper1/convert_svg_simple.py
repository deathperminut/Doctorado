#!/usr/bin/env python3
"""
Script para convertir SVG a PDF usando svglib y reportlab
Método más simple que no requiere cairo
"""
import sys
from pathlib import Path

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
except ImportError:
    print("Instalando dependencias...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "svglib", "reportlab"])
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

# Lista de SVG que necesitamos convertir
svg_files = [
    'distribuciones_parametrosKDM.svg',
    'distribuciones_parametrosTJex2.svg',
    'imagenes_grid.svg',
    'imagenes_grid2.svg',
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

            # Convertir SVG a ReportLab Drawing
            drawing = svg2rlg(str(svg_path))

            if drawing is None:
                raise ValueError("Could not parse SVG")

            # Renderizar como PDF
            renderPDF.drawToFile(drawing, str(pdf_path))

            print(f"  ✓ Created {pdf_path.name}")
            converted += 1

        except Exception as e:
            print(f"  ✗ Failed to convert {svg_file}: {e}")
            failed += 1
    else:
        print(f"  ✗ {svg_file} not found")
        failed += 1

print(f"\nDone! Converted: {converted}, Failed: {failed}")
