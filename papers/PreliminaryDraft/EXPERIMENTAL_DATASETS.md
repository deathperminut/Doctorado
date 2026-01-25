# 📊 Experimental Magnetic Domain Datasets - Comparative Analysis

**Author:** Juan Sebastián Méndez Rondón
**Date:** January 23, 2025
**Purpose:** Doctoral Research - Magnetic Nanodots Characterization

---

## 🎯 Executive Summary

This document provides a comprehensive analysis of **4 experimental datasets** of magnetic domains available for research. The analysis focuses on their applicability to **image-based parameter estimation** and **nanoscale magnetic domain characterization** using deep learning.

### Key Finding
⚠️ **None of the available datasets fully meet the requirements** of having:
- Discrete nanodots (rather than continuous films)
- Complete Hamiltonian parameters (J, K, D, T) associated with images
- Sufficient parameter variation for inverse problem training

**Recommended Strategy:** Hybrid approach using simulations (primary) + experimental datasets (validation/transfer learning)

---

## 📦 Dataset 1: University of Leeds - Skyrmion Hall Angle Dataset

### 🏆 Overall Utility Score: **65%**

### 📋 Basic Information

| Property | Value |
|----------|-------|
| **Experimental Technique** | STXM / MOKE Microscopy |
| **Number of Images** | ~2,400 images |
| **Magnetic Objects** | >1 million skyrmions |
| **Scale** | 35 - 825 nm (diameters) |
| **Dataset Size** | 110 MB (dataset.zip) |
| **Material System** | Ta(3.2)/Pt(2.7)/[Pt(0.6)/CoB(0.8)/Ir(0.4)]×5/Pt(2.2) (nm) |
| **Year** | 2019 |
| **Access** | Open Access |

### 🔬 Available Parameters

| Parameter | Status | Value/Notes |
|-----------|--------|-------------|
| **Keff** (Anisotropy) | ✅ Available | 0.47 ± 0.04 MJ/m³ |
| **Skyrmion Hall Angle** | ✅ Available | 9° ± 2° |
| **Diameters** | ✅ Available | Measured for each skyrmion (35-825 nm) |
| **Velocity** | ✅ Available | 6 ± 1 m/s |
| **DMI (D)** | ⚠️ Implicit | Not reported (but system has interfacial DMI) |
| **Exchange (J/A)** | ❌ Missing | Not reported |
| **Ms** | ❌ Missing | Not reported |
| **Temperature** | ⚠️ Partial | Room temperature only (not varied) |
| **External Field** | ⚠️ Partial | Applied but values in paper |

### ✅ Advantages

1. **Large Dataset:** 2,400+ images with >1M skyrmions
2. **Nanoscale:** Appropriate scale (35-825 nm)
3. **Chiral System:** Implicit DMI from Pt/CoB/Ir interfaces
4. **Size Variability:** Different skyrmion sizes under same conditions
5. **Open Access:** Well-documented and freely available
6. **Topological Textures:** Magnetic skyrmions (topological objects)
7. **High Quality:** Professional STXM/MOKE imaging

### ❌ Limitations

1. **Incomplete Parameters:** Only Keff explicitly reported
2. **Not Discrete Nanodots:** Continuous thin film, not isolated dots
3. **Single Material:** No variation of J, K, D independently
4. **Not for Inverse Problem:** Cannot train parameter estimation (all from same material)
5. **Limited Hamiltonian Info:** Full magnetic parameters not characterized
6. **No Temperature Variation:** Single temperature condition
7. **System-Specific:** Pt/CoB/Ir only - not generalizable

### 🎯 Use Cases for Your Research

| Use Case | Applicability | Description |
|----------|---------------|-------------|
| **Transfer Learning** | ✅ High | Pretrain models to detect magnetic chiral textures |
| **Feature Extraction** | ✅ High | Learn skyrmion representations applicable to nanodots |
| **Qualitative Validation** | ✅ Medium | Compare generated textures vs experimental |
| **Segmentation** | ✅ High | Train detectors for topological magnetic objects |
| **Inverse Problem** | ❌ Low | Cannot estimate Hamiltonian parameters (no variation) |

### 🔗 Links

- **Dataset Repository:** https://archive.researchdata.leeds.ac.uk/619/
- **DOI:** https://doi.org/10.5518/742
- **Paper:** [Nature Communications (2019)](https://www.nature.com/articles/s41467-019-14232-9)
- **Authors:** Zeissler, K. et al. (University of Leeds)

---

## 📦 Dataset 2: Co/Pd Multilayers - MFM Dataset

### 🏆 Overall Utility Score: **60%**

### 📋 Basic Information

| Property | Value |
|----------|-------|
| **Experimental Technique** | MFM (Magnetic Force Microscopy) + AFM |
| **Number of Images** | ~30-50 MFM images |
| **Domain Types** | Stripe, labyrinthine, fragmented domains |
| **Scale** | 100 nm - 5 μm (domain sizes) |
| **Dataset Size** | Supplementary materials |
| **Material System** | Si/Ta(30Å)/Pd(30Å)/[Co(tCo)/Pd(8Å)]×50/Pd(12Å) |
| **Year** | 2022 |
| **Access** | Open Access (Supplementary) |

### 🔬 Available Parameters

| Parameter | Status | Value/Notes |
|-----------|--------|-------------|
| **Keff** (Anisotropy) | ✅ Varies | Changes with ion irradiation |
| **Aex** (Exchange) | ✅ Varies | Changes with ion irradiation |
| **Ms** | ✅ Available | From hysteresis loops |
| **Coercivity** | ✅ Available | Measured from M-H curves |
| **Remanence** | ✅ Available | Measured from M-H curves |
| **Temperature** | ⚠️ Partial | In simulations (300-1000 K), not experiments |
| **Ion Fluence** | ✅ Available | 0.5×10¹⁴ - 3.3×10¹⁶ ions/cm² |
| **Ion Energy** | ✅ Available | 50 and 100 keV |
| **DMI** | ❌ Missing | Not significant in Co/Pd (centrosymmetric) |

### ✅ Advantages

1. **Parameter Variation:** Keff and Aex vary with irradiation conditions
2. **MFM Technique:** Common technique for nanostructures
3. **Simulation Integration:** Includes complementary micromagnetic simulations
4. **Sim-Exp Validation:** Already demonstrates sim-to-exp comparison
5. **Accessible Supplementary:** Data downloadable from publisher
6. **Multiple Conditions:** Different irradiation energies and fluences
7. **Characterized System:** Well-studied Co/Pd multilayers

### ❌ Limitations

1. **Not Nanodots:** Continuous film, not discrete dots
2. **No DMI:** Centrosymmetric system (no Dzyaloshinskii-Moriya interaction)
3. **Inferred Parameters:** Keff, Aex not directly measured, extracted from simulations
4. **Larger Scale:** Micron-scale domains, not <100nm nanodots
5. **Small Dataset:** Only ~30-50 images
6. **Limited Parameter Space:** Only 2 parameters vary independently
7. **Indirect Characterization:** Magnetic parameters from fitting, not direct measurement

### 🎯 Use Cases for Your Research

| Use Case | Applicability | Description |
|----------|---------------|-------------|
| **Domain Adaptation** | ✅ High | Train on sims, adapt to MFM experimental data |
| **Benchmark Testing** | ✅ Medium | Validate models capture stripe/labyrinth domains |
| **Hybrid Methodology** | ✅ High | Learn approach to combine simulations + experiments |
| **Texture Validation** | ✅ Medium | Compare generated domains with real MFM |
| **Limited Inverse** | ⚠️ Low | Only Keff, Aex vary - not full Hamiltonian |

### 🔗 Links

- **Paper:** [ACS Applied Materials & Interfaces (2022)](https://pubs.acs.org/doi/10.1021/acsami.2c12848)
- **Supplementary Data:** https://pubs.acs.org/doi/suppl/10.1021/acsami.2c12848
- **PMC Access:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9650662/
- **Authors:** Kaidatzis, A. et al.

---

## 📦 Dataset 3: KIT RADAR - Lorentz 4D-STEM Dataset

### 🏆 Overall Utility Score: **40%**

### 📋 Basic Information

| Property | Value |
|----------|-------|
| **Experimental Technique** | 4D-STEM Lorentz Microscopy (LA-Ltz-4D-STEM) |
| **Data Type** | Quantitative maps + raw 4D diffraction |
| **Resolution** | <5 nm (nanometric) |
| **Scale** | Nanoscale (strain + magnetic fields) |
| **Dataset Size** | Multiple GB (raw 4D-STEM) |
| **Material System** | Deformed amorphous ferromagnet |
| **Year** | 2025 (Nature Communications - February) |
| **Access** | Open Access |

### 🔬 Available Parameters

| Parameter | Status | Value/Notes |
|-----------|--------|-------------|
| **Magnetic Fields (Bx, By)** | ✅ Available | Pixel-by-pixel vector fields |
| **Strain Tensor** | ✅ Available | Complete strain field maps |
| **Atomic Structure** | ✅ Available | Atomic packing information |
| **Strain-Mag Correlation** | ✅ Available | Pixel-by-pixel correlations |
| **Analysis Scripts** | ✅ Available | MATLAB and Python code |
| **J, K, D, Ms** | ❌ Missing | Hamiltonian parameters not reported |
| **Temperature** | ❌ Missing | Not varied |

### ✅ Advantages

1. **Quantitative Data:** Direct measurement of magnetic fields (not just images)
2. **High Resolution:** <5 nm spatial resolution
3. **Processable Formats:** HDF5, CSV (Python-friendly)
4. **Analysis Scripts Included:** MATLAB/Python for processing
5. **Very Recent:** State-of-the-art (Feb 2025 publication)
6. **Transferable Methodology:** 4D-STEM → magnetic maps reconstruction
7. **Multi-Physical:** Simultaneous magnetic, strain, structure mapping

### ❌ Limitations

1. **Very Different Material:** Amorphous ≠ crystalline nanodots
2. **Focus on Strain:** Not on Hamiltonian parameters (J, K, D)
3. **No Parameter Variation:** Single material system
4. **Different Geometry:** Continuous film, NOT discrete nanodots
5. **Different Physics:** Magnetoelastic coupling vs DMI/exchange/anisotropy
6. **Not Applicable to Direct/Inverse:** System too different from your focus
7. **Specialized Technique:** 4D-STEM not as common as MFM/STM

### 🎯 Use Cases for Your Research

| Use Case | Applicability | Description |
|----------|---------------|-------------|
| **Learn Techniques** | ✅ Medium | Magnetic field reconstruction from TEM |
| **Methodological Validation** | ✅ Medium | Compare simulated vs experimental field maps |
| **Physical Features** | ⚠️ Low | Train with vector fields (not just intensity) |
| **State-of-Art Reference** | ✅ High | Cite in theoretical framework (advanced techniques) |
| **Direct Training** | ❌ Low | System too different for your models |

### 🔗 Links

- **Dataset Repository:** https://radar.kit.edu/radar/en/dataset/ms36bzm0nrrzj51g
- **DOI:** https://doi.org/10.35097/ms36bzm0nrrzj51g
- **Paper:** [Nature Communications (2025)](https://www.nature.com/articles/s41467-025-56521-6)
- **Authors:** Kang, S. et al. (Karlsruhe Institute of Technology)

---

## 📦 Dataset 4: Experimental Exchange Interaction Dataset

### 🏆 Overall Utility Score: **15%**

### 📋 Basic Information

| Property | Value |
|----------|-------|
| **Experimental Technique** | Inelastic Neutron Scattering (INS) |
| **Data Type** | Numerical parameters (NO IMAGES) |
| **Number of Materials** | ~100 magnetic materials |
| **Scale** | Bulk crystals (not nanoscale) |
| **Dataset Size** | GitHub repository + MC files |
| **Material Types** | Various magnetic materials from literature |
| **Year** | 2025 (ArXiv / Nature Scientific Data) |
| **Access** | Open Access (GitHub) |

### 🔬 Available Parameters

| Parameter | Status | Value/Notes |
|-----------|--------|-------------|
| **J (Exchange)** | ✅ Available | Heisenberg model format (unified) |
| **Tc (Curie Temp)** | ✅ Available | Calculated via Monte Carlo |
| **Crystal Structure** | ✅ Available | Visualizations + exchange pathways |
| **Spin Wave Dispersion** | ✅ Available | Experimental INS data |
| **Quantum Correction** | ✅ Available | (S+1)/S factor applied |
| **DMI** | ❌ Excluded | Intentionally excluded from dataset |
| **Domain Images** | ❌ Missing | NO images - only numerical parameters |

### ✅ Advantages

1. **Validated J Parameters:** Experimentally validated exchange constants
2. **Many Materials:** ~100 different magnetic systems
3. **Unified Hamiltonian Format:** Standardized representation
4. **Open Access:** Available on GitHub
5. **Well-Documented Methodology:** Clear extraction procedure
6. **Monte Carlo Simulations:** Includes MC simulation files
7. **Comprehensive Literature:** Compiled from ~100 studies

### ❌ Limitations (CRITICAL)

1. **❌ NO IMAGES:** Only numerical parameters - no domain images
2. **❌ NO DMI:** Explicitly excluded (only Heisenberg exchange)
3. **Bulk Scale:** Large crystals, NOT nanodots
4. **Spin Waves:** Different physics from domain imaging
5. **Not Image-Based:** Cannot use with image-based ML models
6. **Different Experimental Approach:** INS vs MFM/STM/LTEM
7. **Limited Applicability:** Very different from your research focus

### 🎯 Use Cases for Your Research

| Use Case | Applicability | Description |
|----------|---------------|-------------|
| **J Value References** | ⚠️ Low | Validate order of magnitude in simulations |
| **Theoretical Context** | ⚠️ Low | Understand experimental exchange measurement |
| **Methodological Comparison** | ⚠️ Low | INS vs your image-based methods |
| **ML Training** | ❌ None | No images available |
| **DMI Systems** | ❌ None | DMI excluded from dataset |

### 🔗 Links

- **ArXiv Preprint:** https://arxiv.org/abs/2504.15764
- **Nature Scientific Data:** https://www.nature.com/articles/s41597-025-06099-x
- **GitHub Repository:** Check paper for link
- **Authors:** Multiple authors (compilation work)

---

## 📊 Comparative Summary Table

| Dataset | Images | Key Parameters | Scale | Nanodots | Utility |
|---------|--------|----------------|-------|----------|---------|
| **Leeds Skyrmions** | ✅✅✅ ~2400 | ⚠️ Keff (DMI implicit) | ✅ 35-825nm | ⚠️ Continuous film | **65%** |
| **Co/Pd MFM** | ✅ ~30-50 | ✅ Keff, Aex vary | ⚠️ 100nm-5μm | ❌ Continuous film | **60%** |
| **KIT 4D-STEM** | ✅ Quantitative maps | ⚠️ Fields, strain | ✅ <5nm | ❌ Amorphous film | **40%** |
| **Exchange Dataset** | ❌ None | ✅ J (no DMI) | ❌ Bulk | ❌ No | **15%** |

### Legend
- ✅ = Excellent/Available
- ⚠️ = Partial/Limited
- ❌ = Missing/Not applicable

---

## 🎯 Recommendations for Your Research

### Priority Ranking

1. **🥇 Leeds Skyrmions Dataset (65% utility)**
   - **Best for:** Transfer learning, feature extraction, chiral texture recognition
   - **Use case:** Pretrain models on large skyrmion dataset, then fine-tune on your nanodots
   - **Download priority:** HIGH

2. **🥈 Co/Pd MFM Dataset (60% utility)**
   - **Best for:** Sim-to-exp domain adaptation, validation methodology
   - **Use case:** Learn how to bridge simulation and experimental data gaps
   - **Download priority:** MEDIUM-HIGH

3. **🥉 KIT 4D-STEM Dataset (40% utility)**
   - **Best for:** Advanced methodology, quantitative field reconstruction
   - **Use case:** Reference for state-of-the-art techniques, cite in literature review
   - **Download priority:** LOW (study paper first)

4. **❌ Exchange Dataset (15% utility)**
   - **Best for:** Theoretical reference only
   - **Use case:** Validate J values in your simulations
   - **Download priority:** VERY LOW

---

## 💡 Recommended Strategy: Hybrid Approach

### The Critical Gap

⚠️ **None of the available experimental datasets provide:**
- Discrete nanodots with controlled geometries
- Complete set of Hamiltonian parameters (J, K, D, T) associated with each image
- Sufficient parameter variation for training inverse models

### Proposed Solution: Multi-Level Validation

```
┌─────────────────────────────────────────────────┐
│   PRIMARY: Your Simulations                     │
│   - Nanodots with J, K, D, T variation         │
│   - Complete ground truth                       │
│   - Large dataset (100k+ images possible)       │
└────────────────┬────────────────────────────────┘
                 │
                 ├──► Train base models
                 │
┌────────────────▼────────────────────────────────┐
│   VALIDATION LAYER 1: Transfer Learning         │
│   - Leeds Skyrmions (2400 images)              │
│   - Pretrain feature extractors                 │
│   - Learn chiral texture representations         │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│   VALIDATION LAYER 2: Domain Adaptation         │
│   - Co/Pd MFM (30-50 images)                   │
│   - Bridge sim-to-real gap                      │
│   - Validate domain texture generation           │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│   VALIDATION LAYER 3: Methodology               │
│   - KIT 4D-STEM (quantitative fields)          │
│   - Validate magnetic field reconstruction       │
│   - Compare with state-of-the-art               │
└─────────────────────────────────────────────────┘
```

### For Your Paper/Thesis

**State the limitation honestly:**
> "While experimental datasets of magnetic nanodots with complete Hamiltonian parameter characterization are not publicly available, we validate our framework through a multi-level strategy: (1) extensive testing on Hamiltonian-derived simulations with known ground truth, (2) transfer learning validation on experimental skyrmion datasets [Leeds], (3) domain adaptation to MFM experimental data [Co/Pd], and (4) methodological comparison with state-of-the-art quantitative imaging [KIT 4D-STEM]."

**Highlight the contribution:**
> "Our physics-informed framework addresses a critical gap: existing experimental datasets either lack associated Hamiltonian parameters or focus on continuous films rather than discrete nanodots. This work demonstrates how deep learning can bridge this gap by learning from simulations while maintaining physical interpretability and experimental transferability."

---

## 📥 Next Steps

### Immediate Actions

1. **Download Leeds Skyrmions Dataset**
   - Size: 110 MB (manageable)
   - Extract and explore structure
   - Plan transfer learning experiments

2. **Access Co/Pd Supplementary Materials**
   - Download MFM images
   - Study their sim-exp methodology
   - Plan domain adaptation approach

3. **Read Recent Papers**
   - KIT 4D-STEM paper (Feb 2025) - state of the art
   - Leeds skyrmion paper - understand their system
   - Co/Pd paper - learn sim-exp validation

### Medium-Term Actions

1. **Search for Additional Datasets**
   - Contact authors of recent papers directly
   - Check for supplementary data in skyrmion/nanodot papers 2023-2025
   - Explore Materials Project, NOMAD repository with specific queries

2. **Consider Collaboration**
   - Identify experimental groups working with nanodots
   - Propose collaboration for dataset generation
   - Could lead to high-impact publication with joint dataset

3. **Document Limitation in Proposal**
   - Clearly state lack of ideal experimental dataset
   - Justify hybrid simulation + validation approach
   - Emphasize novel contribution: creating methodology for this gap

---

## 📚 Citation Information

If you use this analysis or these datasets, consider citing:

**Leeds Skyrmions:**
```
Zeissler, K. et al. (2019). Diameter-independent skyrmion Hall angle
observed in chiral magnetic multilayers. Nature Communications, 10(1), 4862.
DOI: 10.1038/s41467-019-14232-9
```

**Co/Pd MFM:**
```
Kaidatzis, A. et al. (2022). Understanding the Magnetic Microstructure
through Experiments and Machine Learning Algorithms.
ACS Applied Materials & Interfaces. DOI: 10.1021/acsami.2c12848
```

**KIT 4D-STEM:**
```
Kang, S. et al. (2025). Large-angle Lorentz Four-dimensional scanning
transmission electron microscopy for simultaneous local magnetization,
strain and structure mapping. Nature Communications.
DOI: 10.1038/s41467-025-56521-6
```

**Exchange Dataset:**
```
Authors (2025). Experimental Exchange Interaction Dataset for Magnetic
Materials: Spin Waves to MC Simulations. Nature Scientific Data.
DOI: 10.1038/s41597-025-06099-x
```

---

## 📧 Contact & Updates

For questions or updates to this analysis:
- **Author:** Juan Sebastián Méndez Rondón
- **Date:** January 23, 2025
- **Version:** 1.0

---

**Document End** | Generated for Doctoral Research - Magnetic Nanodots Characterization using Physics-Informed Deep Learning