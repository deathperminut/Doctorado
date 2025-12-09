#!/usr/bin/env python3
"""
Script to reconvert cropped figures using svglib + reportlab
This method respects the full SVG viewBox
"""
import sys
from pathlib import Path

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
except ImportError:
    print("Installing dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "svglib", "reportlab"])
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

# List of SVG files that need to be reconverted
svg_files = [
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

            # Convert SVG to ReportLab Drawing
            drawing = svg2rlg(str(svg_path))

            if drawing is None:
                raise ValueError("Could not parse SVG")

            # Render as PDF
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
