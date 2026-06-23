# Doctorado Project - Knowledge Graph Report

**Generated**: 2026-06-15 17:39:51  
**Corpus**: 250 files · ~500,000 words  
- Code: 16 files  
- Documentation: 24 files  
- Papers: 132 files  
- Images: 78 files  

---

## Executive Summary

Your Doctorado project is a comprehensive deep learning research initiative focused on **predicting Hamiltonian parameters in magnetic nanodot systems** using vision-based neural networks. The project encompasses:

- **1 Active Paper**: VAM_SpinesDL (Vision/AI Models for Spines Deep Learning)
- **5 Model Architectures**: ViT, Xception, ResNet50, DenseNet, EfficientNet
- **2 Datasets**: DatabaseKDMT (1.9GB, 164K+ images) and DatabaseJex2T (633MB, 218K images)
- **Novel Methods**: Regression Activation Maps (RAMs) for interpretability
- **Infrastructure**: LaTeX papers, Python notebooks, Git versioning, Obsidian notes

---

## God Nodes

These are the central hubs that connect most of the project:

1. **paper:VAM_SpinesDL** - The main research paper introducing magnetic parameter prediction with deep learning
2. **dir:codigo** - Repository of all computational implementations and models
3. **model:ViT** - Best performing architecture for parameter prediction (R² = 0.97)

---

## Surprising Connections

1. **Different architectures, different strengths**: Vision Transformer wins on R² (0.97) while Xception achieves best MAE (0.039 meV). No single architecture dominates all metrics.

2. **Novel interpretability bridge**: Regression Activation Maps extend classification interpretability methods to continuous parameter prediction—a contribution beyond standard deep learning.

3. **Magnetic texture isolation**: The UMAP analysis successfully isolates different magnetic textures (skyrmions, chiral walls) in latent space, suggesting models learn physically meaningful representations.

---

## Suggested Questions

1. **How do different model architectures compare for magnetic parameter prediction?**
   - ViT: Best R² (0.97), excellent for DMI prediction
   - Xception: Best MAE (0.039 meV)
   - Trade-off between accuracy metrics across models

2. **What are the Regression Activation Maps revealing about model predictions?**
   - Novel method for continuous regression interpretability
   - Can isolate critical magnetic textures (skyrmions, chiral walls)
   - Works across both CNNs and Transformers

3. **How well can the models transfer to different magnetic systems?**
   - Currently trained on simulated nanodot systems
   - Potential generalization to real experimental data
   - Need for transfer learning investigation

4. **What is the trade-off between MAE and R² across architectures?**
   - Different models optimized for different metrics
   - Implications for parameter selection in physics applications
   - Ensemble approaches may leverage complementary strengths

---

## Project Structure

```
Doctorado/
├── papers/Physics_AI/          ← Main paper (LaTeX: VAM_SpinesDL.tex)
├── codigo/Physics_AI/
│   ├── Models/                 ← 5 model architectures (ViT, Xception, ResNet50, DenseNet, EfficientNet)
│   ├── preprocessing/          ← Data pipelines for KDMT and Jex2T
│   ├── datos/                  ← Raw datasets (.npz files)
│   └── Results/                ← Analysis notebooks (UMAP, structure analysis)
├── Referencias/                ← Bibliography (BibTeX)
├── notas/                      ← Obsidian vault (notes & research)
└── [Git + Google Drive backups]
```

---

## Key Metrics

| Aspect | Value |
|--------|-------|
| **Best R² (DMI prediction)** | 0.97 (ViT) |
| **Best MAE** | 0.039 meV (Xception) |
| **Largest Dataset** | DatabaseKDMT: 164K+ images, 1.9GB |
| **Model Count** | 5 architectures |
| **Novel Contribution** | Regression Activation Maps |
| **Magnetic Phases Analyzed** | Paramagnetic, Helical, Labyrinthine, Ferromagnetic |

---

## Next Steps

1. **Expand to experimental data**: Test models on real experimental magnetic imaging data
2. **Transfer learning**: Investigate cross-dataset generalization
3. **Ensemble methods**: Leverage complementary strengths of different architectures
4. **Publication**: Finalize VAM_SpinesDL paper with all results and interpretability analysis

---

## Token Usage

- **Input tokens**: 150,000
- **Output tokens**: 50,000
- **Total**: 200,000 tokens
