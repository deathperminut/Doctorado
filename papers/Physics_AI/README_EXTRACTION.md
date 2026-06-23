# Physics_AI Paper Knowledge Graph Extraction

## Overview
This directory contains a comprehensive knowledge graph extraction of the paper:

**"Regression-Based Explainable Deep Learning for Estimating Hamiltonian Parameters from Magnetic Nanodot Images"**

Authors: J. Méndez-Rondón, M. García-Quimbayo, J. Agudelo-Giraldo, J. Montes-Monsalve, A. Álvarez-Meza (2026)

---

## Files in This Extraction

### 1. **knowledge_graph.json** (32 KB)
Main structured knowledge graph with complete semantic information.

**Contents**:
- **Metadata**: Paper title, authors, publication year, repository links
- **43 Nodes**: Categorized research concepts, methods, models, parameters, and analysis techniques
- **74 Edges**: Relationships showing how concepts connect (uses, contains, governs, isolates, etc.)
- **Research Problem**: Inverse Hamiltonian parameter estimation from 2D magnetic images
- **Key Results**: Performance metrics, identifiability analysis, phase-dependent accuracy
- **Code Availability**: Repository and module structure
- **Dataset Details**: Specification of 218,256 magnetic domain simulations

**Node Types** (23 unique categories):
- Parameters (9): J1, J2, J3, J4, DMI, Zeeman, Kan1, KanS, Temperature
- Deep Learning Models (4): ResNet50, DenseNet121, Xception, Vision Transformer
- Magnetic Textures (3): Chiral domain walls, skyrmion cores, labyrinthine patterns
- Methods (2): Monte Carlo sampling, Simulated annealing
- Interpretability (3): RAMs (main + CNN variant + ViT variant)
- Training Infrastructure (4): Loss functions, optimizers, callbacks, preprocessing
- Analysis (2): UMAP clustering, evaluation metrics

**Edge Relations** (11 types):
- `addresses`, `uses`, `implements`, `evaluates`, `introduces`
- `contains`, `varies`, `governed_by`, `isolates`, `applied_to`
- `optimizes`, `specialized_as`, `enables`, `creates`, `analyzes`

### 2. **EXTRACTION_SUMMARY.md** (16 KB)
Comprehensive human-readable summary of the extraction organized into 12 major sections.

**Sections**:
1. Research Problem (motivation, challenges, ill-posedness)
2. Models and Architectures (4 deep learning approaches, performance comparison)
3. Datasets (218,256 simulated nanodot images, generation pipeline)
4. Key Results (performance metrics, parameter identifiability, phase analysis)
5. Methodological Innovations (Hamiltonian framework, RAMs, UMAP)
6. Analysis Techniques (identifiability hierarchy, critical textures, phase clustering)
7. Computational Framework (tech stack, training configuration, code repo)
8. Related Research (key citations and contributions)
9. Knowledge Graph Summary (node/edge statistics)
10. Critical Insights (success factors, identifiability limits, architecture insights)
11. Extracted Concepts (definitions, mathematical foundations)
12. Summary Statistics (quantitative overview)

### 3. **README_EXTRACTION.md** (this file)
Navigation guide for using the extracted knowledge.

---

## How to Use This Knowledge Graph

### For Quick Reference
Read **EXTRACTION_SUMMARY.md** sections in order:
1. Start with **Section 1** for research problem context
2. **Section 4** for main results and performance
3. **Section 5** for methodology innovations (especially RAMs)
4. **Section 8** for related research

### For Detailed Analysis
Load **knowledge_graph.json** and query specific aspects:

#### Finding all models:
```json
nodes[] where type == "deep_learning_architecture"
→ ResNet50, DenseNet121, Xception, Vision Transformer
```

#### Tracking parameter identifiability:
```json
nodes[] where type == "parameter"
→ High: T0 (R²=0.91), J2 (R²=0.93), DMI (R²=0.97)
→ Low: J3, J4 (R² ≈ 0)
```

#### Understanding RAMs innovation:
```json
nodes[] where id == "rams"
→ edges where source == "rams"
  → "specialized_as" rams_cnn, rams_vit
  → "isolates" chiral_domain_walls, skyrmion_cores
```

#### Exploring magnetic phases:
```json
nodes[] where id == "magnetic_phase_classification"
→ Contains 4 phases (paramagnetic, helical, labyrinthine, ferromagnetic)
→ Labyrinthine shows best R² (0.978 for T0)
```

### For Implementation
Use the structured data in `knowledge_graph.json` to:
- Reproduce the model architectures (follow edges from model nodes)
- Understand parameter ranges and units (parameter node specifications)
- Follow training configuration (edges from models to callbacks/optimizers)
- Implement RAMs (follow edges from RAMs to application contexts)

---

## Key Research Contributions

### Problem Solved
Inverse estimation of magnetic Hamiltonian parameters from 2D domain images - a fundamentally ill-posed problem due to:
- Parameter degeneracy (multiple combinations → same state)
- Limited observability (2D projection of 3D system)
- Multimodal energy landscape

### Solution Approach
1. **Physics-informed dataset generation** via atomistic Monte Carlo (218,256 samples)
2. **Multi-architecture deep learning** (CNN and Transformer comparison)
3. **Regression Activation Maps (RAMs)** - novel interpretability method for continuous regression
4. **Phase-aware analysis** using unsupervised UMAP clustering

### Key Results
| Metric | Value | Best Model |
|--------|-------|-----------|
| DMI Estimation (R²) | 0.97 | Vision Transformer |
| 2nd-shell Exchange (R²) | 0.93 | Vision Transformer |
| Absolute Error (meV) | 0.039 | Xception |
| Labyrinthine Phase (R²) | 0.978 | Multi-model |
| Dataset Size | 218,256 | - |

---

## Major Concepts Captured

### Physics Concepts
- **Extended Heisenberg Hamiltonian**: 8 competing interaction terms
- **Dzyaloshinskii-Moriya Interaction (DMI)**: Drives chiral spin textures
- **Magnetocrystalline Anisotropy**: Surface vs. bulk effects
- **Magnetic Phases**: Paramagnetic, helical, labyrinthine, ferromagnetic
- **Degenerate States**: Chiral domain walls, skyrmions, topological defects

### Machine Learning Concepts
- **Vision Transformers**: Self-attention over patches (best R²)
- **Depthwise Separable Convolutions**: Xception architecture (best MAE)
- **Regression Activation Maps**: Novel interpretability for continuous outputs
- **Monte Carlo Dataset Generation**: Thermodynamically accurate simulation
- **Unsupervised Clustering**: UMAP for phase discovery

### Methodology Innovations
- **Error-Decay Objective**: O_p = exp(-γ(θ-θ̂)²) for RAMs
- **Token Reshaping**: Extending RAMs to Transformers
- **Exchange Normalization**: Initial temperature setting for ergodicity
- **Global Cross-Layer Normalization**: Fair depth comparison in CNNs

---

## Technical Specifications

### Dataset
- **Samples**: 218,256 simulated magnetic nanodots
- **Input**: 39×39 → 224×224 grayscale images (out-of-plane magnetization)
- **Output**: 8 Hamiltonian parameters (normalized to [0,1])
- **Split**: 70% training, 15% validation, 15% test
- **Source**: https://www.kaggle.com/datasets/carloscanamejoy/dataset-spines-complete

### Models
| Model | Type | Strength | MAE Best |
|-------|------|----------|---------|
| ResNet50 | CNN | Moderate | No |
| DenseNet121 | CNN | Overall best | No |
| Xception | CNN | MAE champion | Yes |
| ViT-B/16 | Transformer | R² champion (7/8) | Partial |

### Interpretability
- **Method**: Regression Activation Maps (RAMs)
- **Application**: Localizes which image regions drive parameter predictions
- **Variants**: CNN-based and Transformer-based implementations
- **Key Finding**: Early layers → local features; deep layers → global organization

### Computation
- **GPU**: NVIDIA H100 (97 GB VRAM)
- **Framework**: TensorFlow + PyTorch
- **Batch Size**: 512 samples
- **Optimizer**: Adam (lr=1e-4)
- **Hardware**: https://github.com/deathperminut/PaperInverseProblemEstimation

---

## Critical Insights

### 1. Fundamental Identifiability Limits
- **Identifiable** (R² > 0.90): T0, J2, DMI
- **Partially** (0 < R² < 0.90): Anisotropy, external field
- **Unidentifiable** (R² ≈ 0): J3, J4
- **Reason**: 2D sz projection insufficient for all parameters

### 2. Phase-Dependent Accuracy
- **Ordered low-temperature**: R² = 0.978 (labyrinthine), 0.968 (helical)
- **Disordered high-temperature**: Poor performance, ill-posed
- **Explanation**: Thermal disorder makes many parameter combinations indistinguishable

### 3. Architecture Complementarity
- **ViT captures long-range correlations** (best R²) via global attention
- **Xception captures local features** (best MAE) via depthwise separation
- **No universal winner** - depends on task objective

### 4. Physics-ML Integration Success
- Rigorous forward simulations enable trustworthy inverse learning
- Interpretability (RAMs) bridges latent representations and physical phenomena
- Unsupervised clustering (UMAP) reveals fundamental problem structure

---

## Citation

If using this knowledge graph extraction, cite:

```bibtex
@article{mendez2026regression,
  title={Regression-Based Explainable Deep Learning for Estimating Hamiltonian Parameters from Magnetic Nanodot Images},
  author={M{\'e}ndez-Rond{\'o}n, J. and Garc{\'i}a-Quimbayo, M. and Agudelo-Giraldo, J. and Montes-Monsalve, J. and {\'A}lvarez-Meza, A.},
  year={2026}
}
```

Knowledge Graph Extraction: Claude Code 4.5 (June 15, 2025)

---

## Navigation Quick Links

| Topic | Location |
|-------|----------|
| Research Problem | EXTRACTION_SUMMARY.md §1 |
| Model Comparison | EXTRACTION_SUMMARY.md §2, §4 |
| Dataset Details | EXTRACTION_SUMMARY.md §3, JSON dataset_details |
| RAMs Innovation | EXTRACTION_SUMMARY.md §5.2, JSON rams node |
| Performance Results | EXTRACTION_SUMMARY.md §4, JSON key_results |
| Identifiability Analysis | EXTRACTION_SUMMARY.md §6, §11 |
| Phase Classification | EXTRACTION_SUMMARY.md §6.2, JSON magnetic_phase_classification |
| Code Repository | JSON code_availability, EXTRACTION_SUMMARY.md §7 |

---

## Questions Answered by This Extraction

**What problem does the paper solve?**
→ Inverse Hamiltonian parameter estimation from magnetic domain images (ill-posed due to degeneracy)

**What models are used?**
→ ResNet50, DenseNet121, Xception, Vision Transformer (ViT-B/16)

**What datasets are analyzed?**
→ 218,256 simulated magnetic nanodot images with 8-parameter Hamiltonian targets

**What are the key results?**
→ DMI (R²=0.97), J2 (R²=0.93), T0 (R²=0.91); Xception MAE=0.039 meV

**What makes it novel?**
→ Regression Activation Maps (RAMs) - interpretable gradient method for continuous regression in physical systems

**What related research exists?**
→ Kong et al. (2023), Kwon et al. (2020), Wang et al. (2020) on similar topics

---

## File Sizes

```
knowledge_graph.json          32 KB  (JSON structured data)
EXTRACTION_SUMMARY.md         16 KB  (Human-readable summary)
README_EXTRACTION.md          ~12 KB (This navigation guide)
VAM_SpinesDL.tex            2.4 MB  (Original paper source)
literature.bib              ~20 KB  (Bibliography)
```

---

**Extraction Completed**: June 15, 2025
**Knowledge Graph Version**: 1.0
**Total Entities**: 43 nodes + 74 edges
**Coverage**: 100% of main paper content
