#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

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

for svg_file in svg_files:
    svg_path = figures_dir / svg_file
    pdf_path = figures_dir / svg_file.replace('.svg', '.pdf')

    if svg_path.exists():
        print(f"Converting {svg_file}...")
        # Usar qlmanage con mejor configuración
        cmd = f'qlmanage -t -s 3000 -o "{figures_dir}" "{svg_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True)

        # qlmanage crea archivos con .svg.png, necesitamos renombrarlos
        thumb_path = figures_dir / f"{svg_file}.png"
        if thumb_path.exists():
            # Convertir PNG a PDF usando sips
            cmd2 = f'sips -s format pdf "{thumb_path}" --out "{pdf_path}"'
            subprocess.run(cmd2, shell=True, capture_output=True)
            thumb_path.unlink()  # Eliminar el PNG temporal
            print(f"  ✓ Created {pdf_path.name}")
        else:
            print(f"  ✗ Failed to create thumbnail for {svg_file}")
    else:
        print(f"  ✗ {svg_file} not found")

print("\nDone!")
