# Physics_AI Paper Knowledge Graph Extraction Summary

## Document Information
- **Paper Title**: Regression-Based Explainable Deep Learning for Estimating Hamiltonian Parameters from Magnetic Nanodot Images
- **Authors**: J. Méndez-Rondón, M. García-Quimbayo, J. Agudelo-Giraldo, J. Montes-Monsalve, A. Álvarez-Meza
- **Year**: 2026
- **Field**: Computational Physics, Machine Learning, Condensed Matter Physics

## 1. Research Problem

### Core Challenge
Accurately inferring atomistic magnetic parameters from 2D magnetic domain visualizations of nanodots. This is an **ill-posed inverse problem** characterized by:

- **Structural Degeneracy**: Multiple Hamiltonian parameter combinations produce indistinguishable magnetic states
- **Limited Observability**: Only 2D projection (s_z) available, missing in-plane components (s_x, s_y)
- **Multimodality**: Highly multimodal energy landscape with competing interactions
- **Noise Sensitivity**: Experimental data subject to measurement distortions and finite resolution

### Motivation
Understanding nanoscale magnetic behavior is crucial for:
- Spintronic devices development
- Ultrahigh-density magnetic memories
- Neuromorphic computing platforms based on topological textures
- Materials engineering and design optimization

---

## 2. Models and Architectures

### Deep Learning Architectures Evaluated

#### **ResNet50** (Residual Networks)
- **Type**: Convolutional Neural Network with residual shortcuts
- **Key Mechanism**: Skip connections bypass multiple layers, mitigating vanishing gradients
- **Advantage**: Enables very deep networks while preserving spatial information
- **Performance**: Moderate; competitive but not best overall

#### **DenseNet121** (Densely Connected Networks)
- **Type**: Convolutional network with dense feature concatenation
- **Key Mechanism**: Features concatenated across all preceding layers
- **Advantage**: Maximizes information flow and multi-scale analysis
- **Performance**: Strongest overall in preliminary results
- **Suited for**: Complex overlapping spatial frequencies

#### **Xception** (Extreme Inception)
- **Type**: Depthwise separable convolutional network
- **Key Mechanism**: Decouples spatial correlations (depthwise) from cross-channel correlations (pointwise)
- **Advantage**: Computational efficiency with high representational power
- **Performance**: 
  - **Best MAE** (lowest absolute errors)
  - Best for parameters: T0, J2, J3, J4, H_ex, K_DM
  - MAE values: ~0.039 meV for J2 and K_DM
- **Suited for**: Local topological feature extraction

#### **Vision Transformer (ViT-B/16)**
- **Type**: Patch-based transformer with self-attention
- **Key Mechanism**: Multi-head self-attention over non-overlapping patch embeddings
- **Advantage**: Captures long-range global spatial dependencies instantly
- **Performance**:
  - **Best R² scores** (explains most variance)
  - Best for 7 of 8 parameters (T0, J3, J4, K_an1, K_anS, H_ex, K_DM)
  - Only best MAE for anisotropy constants
- **Suited for**: Global topological stability analysis

### Complementary Strengths
- **ViT**: Superior variance explanation (R²) → better for global correlations
- **Xception**: Tighter absolute predictions (MAE) → better for dense ordered regions
- **DenseNet**: Best overall balance in preliminary results

---

## 3. Datasets

### Main Dataset: Simulated Magnetic Nanodots

#### Specification
- **Total Samples**: 218,256
- **Input**: Grayscale magnetic domain images (39×39 native, 224×224 processed)
- **Output**: 8-dimensional Hamiltonian parameter vectors

#### Nanodot Geometry
- **Shape**: Cylindrical disk
- **Radius**: 18.25 Magnetic Unit Cells (MUC)
- **Thickness**: 5 MUC
- **Total Spins**: ~5,200 localized magnetic moments
- **Lattice**: 3D discrete cubic lattice
- **Projection**: Central layer z = L/2 (2D s_z projection)

#### Generation Method
1. **Atomistic Simulation**: Extended Heisenberg Hamiltonian
2. **Thermal Relaxation**: Metropolis-Hastings Monte Carlo + Simulated Annealing
3. **Parameter Sampling**: Uniform random sampling across 8D parameter space
4. **Image Rendering**: Divergent red-blue colormap encoding spin polarities
5. **Data Split**: 70% training, 15% validation, 15% testing

#### Availability
**URL**: https://www.kaggle.com/datasets/carloscanamejoy/dataset-spines-complete

---

## 4. Key Results and Performance Metrics

### Overall Performance Summary

| Metric | Best Model | Value |
|--------|-----------|-------|
| **R² Variance Explained** | Vision Transformer | 0.91-0.97 |
| **MAE Lowest Error** | Xception | 0.039 meV |
| **Parameters Predicted** | All models | 8 parameters |
| **Dataset Samples** | - | 218,256 |

### Parameter-Specific Performance

#### High Identifiability Parameters
- **T0 (Annealing Temperature)**: R² = 0.91 | Controls global spin order
- **J2 (2nd-shell Exchange)**: R² = 0.93, MAE = 0.039 meV | Modulates domain collinearity
- **K_DM (DMI)**: R² = 0.97, MAE = 0.039 meV | Dictates wavelength and chirality

#### Intermediate Identifiability
- **K_an1** (Bulk Anisotropy): Partially identifiable
- **K_anS** (Surface Anisotropy): Partially identifiable
- **H_ex** (External Field): Partially identifiable

#### Low Identifiability Parameters
- **J3 (3rd-shell Exchange)**: R² ≈ 0 | Energetically dominated by J1, J2
- **J4 (4th-shell Exchange)**: R² ≈ 0 | Weak spatial signature

### Phase-Dependent Performance

#### Labyrinthine & Conical Phase (Ordered Low-Temperature)
- **R² for T0**: 0.978 ✓ Highest accuracy
- **R² for K_DM**: 0.970 ✓ Highest accuracy
- **Characteristics**: Clear stripe-like domains with chiral boundaries

#### Helical Phase (Ordered Low-Temperature)
- **R² for T0**: 0.968 ✓ Good performance
- **R² for K_DM**: 0.866 | Good performance
- **Characteristics**: Spiral spin textures with long-range order

#### Paramagnetic Phase (High-Temperature Disorder)
- **Performance**: Poor
- **Bulk Error Concentration**: ~60% of residual error
- **Issue**: Thermally disordered, visually indistinguishable configurations

### Critical Finding
The apparent contradiction between high R² and larger MAE for T0, J2, K_DM is NOT a modeling failure but rather reflects the **fundamental ill-posedness of the inverse problem** in high-temperature disordered regimes where thermal fluctuations collapse distinct parameter combinations into identical textures.

---

## 5. Methodological Innovations

### 1. Physics-Informed Dataset Generation

#### Extended Heisenberg Hamiltonian Framework
The Hamiltonian explicitly captures:
```
H = - Σ J_r ⟨s_i, s_j⟩           [Symmetric exchange]
    + D ⟨d_ij, (s_i × s_j)⟩     [DMI interaction]
    - H Σ ⟨e_x, s_i⟩             [Zeeman coupling]
    + K_i (quartic anisotropy)    [Magnetocrystalline anisotropy]
```

#### Thermodynamic Relaxation via Monte Carlo
- **Algorithm**: Metropolis-Hastings with local spin perturbations
- **Cooling Schedule**: Linear decrement (0.1 K per step)
- **Initial Temperature**: Exchange-normalized to overcome local minima
- **Thermalization**: 8,000 MCS per temperature step
- **Measurement**: 100 MCS per temperature step

#### Parameter Ranges
| Parameter | Min | Max | Unit | Role |
|-----------|-----|-----|------|------|
| T⁽⁰⁾ | 0.2 | 20.0 | K | Annealing temperature |
| J₁ | 1.0 | 1.0 | meV | Reference energy (fixed) |
| J₂ | -0.66 | 0.66 | meV | 2nd-shell exchange |
| J₃ | -0.29 | 0.29 | meV | 3rd-shell exchange |
| J₄ | -0.23 | 0.23 | meV | 4th-shell exchange |
| D/K_DM | 0.0 | 1.2 | meV | DMI strength |
| H_ex | 0.0 | 0.05 | meV/atom | External field |
| K_an1 | 0.0 | 0.2 | meV/atom | Bulk anisotropy |
| K_anS | 0.0 | 0.2 | meV/atom | Surface anisotropy |

### 2. Regression Activation Maps (RAMs) - Novel Interpretability Method

#### Problem Addressed
Standard gradient-based activation methods (Grad-CAM) fail for continuous regression because:
- No discrete target class to isolate output dimension
- Gradient instability from raw prediction values
- Loss of physical consistency in multimodal systems

#### RAMs Solution
**Error-Decay Objective Function**:
```
O_p = exp(-γ_p (θ_p - θ̂_p)²)
```

Where:
- γ_p = 10 (precision factor)
- θ_p = ground truth parameter
- θ̂_p = predicted parameter
- Output: Smooth activation in (0, 1]

**Spatial Importance Weights**:
```
α_k^(p) = (1/H'W') Σ ∂O_p/∂A_ij^k
```

**Final RAM**:
```
R_RAM^(p) = ReLU(Σ_k α_k^(p) A^k)
```

#### Key Advantages
1. **Continuous mapping**: Directly correlates spatial features to physical parameter errors
2. **Physically traceable**: Links latent representations to observed magnetic phenomena
3. **Architecture-agnostic**: Works for both CNNs and Transformers
4. **Isolates degenerate textures**: Identifies critical features (domain walls, skyrmions)

#### CNN vs. Transformer Implementations

**CNN Variant**:
- Operates on terminal convolutional feature maps
- Straightforward spatial gradient computation
- Shows hierarchical depth-wise decoupling

**ViT Variant**:
- Requires token reshaping: (N_r × D) → (√N_r × √N_r × D)
- Maps sequence embeddings back to 2D spatial grid
- Shows globally coherent, patch-level attribution

### 3. Magnetic Phase Classification via UMAP

Unsupervised clustering identifies **4 physically meaningful phases**:

1. **Paramagnetic Phase**: High-temp disorder (error concentration)
2. **Helical Phase**: Chiral spiral winding (R² = 0.968 for T0)
3. **Labyrinthine & Conical**: Domain walls (R² = 0.978 for T0)
4. **Ferromagnetic/Mixed**: Variable performance

**Key Insight**: Unsupervised clustering reveals that fundamental ill-posedness concentrates in high-T paramagnetic regime, confirming physical rather than algorithmic limitations.

---

## 6. Analysis Techniques and Results

### Identifiability Hierarchy (Discovered via RAMs)

**Early Layers (Convolutional Backbone)**:
- High activation for short-range parameters: J₁, J₂, D
- Capture localized chiral winding angles
- Focus on domain wall boundaries

**Intermediate Layers**:
- Aggregate local features into broader topological patterns
- Intermediate integration of short and long-range effects

**Deep Layers (Terminal Backbone)**:
- High activation for long-range parameters: H_ex, K_an1, K_anS
- Encode global topological confinement
- Resolve macro-scale phase transitions

### Critical Magnetic Textures Isolated by RAMs

1. **Chiral Domain Walls**
   - Degenerate topological features with non-collinear spin winding
   - Governed by DMI and exchange competition
   - Located in early layers

2. **Skyrmion Cores**
   - Topologically protected spin defects
   - Critical for spintronic applications
   - Detected by intermediate layers

3. **Labyrinthine Patterns**
   - Stripe-like domains with regular spacing
   - Characteristic of helical phase
   - High R² = 0.978 performance

### UMAP-Based Phase Space Navigation

- **Unsupervised Learning**: No labels required
- **Dimensionality Reduction**: Projects high-dimensional latent space to 2D
- **Reveals**: Natural clustering into 4 magnetic phase categories
- **Validates**: Ill-posedness is concentrated in specific paramagnetic regions

---

## 7. Computational Framework

### Technology Stack
- **Language**: Python 3
- **Deep Learning**: TensorFlow/Keras
- **Physics Simulations**: PyTorch with CUDA
- **Computing**: NVIDIA H100 GPU (97 GB VRAM)

### Training Configuration
- **Batch Size**: 512 samples
- **Max Epochs**: 100
- **Optimizer**: Adam (lr = 1e-4)
- **Loss Function**: Mean Squared Error (MSE)
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.3, patience=4)
- **Early Stopping**: Patience = 8 epochs with best weight restoration

### Regularization
- Batch Normalization (before/after dense layers)
- Dropout (p=0.4, p=0.3)
- L₂ weight regularization (λ=1e-4)
- Min-Max normalization of targets to [0, 1]

### Code Availability
**GitHub**: https://github.com/deathperminut/PaperInverseProblemEstimation

**Modules**:
- `/simulation`: Atomistic Monte Carlo scripts (PyTorch + CUDA)
- `/regression`: Training/validation/evaluation pipelines (TensorFlow)
- `/rams`: RAMs computation and visualization (Xception + ViT)

---

## 8. Related Research and Contributions

### Key Related Works Cited
1. **Kong et al. (2023)**: Quantifying magnetic interactions via deep neural networks
2. **Kwon et al. (2020)**: Magnetic Hamiltonian parameter estimation using deep learning
3. **Wang et al. (2020)**: Learning magnetic parameters from spin configurations
4. **Fert et al. (2017)**: Foundational work on magnetic skyrmions

### Novel Contributions of This Work
1. **First explainable regression framework** for continuous Hamiltonian parameter estimation
2. **RAMs method**: Bridges interpretability gap between black-box models and physical laws
3. **Physics-first approach**: Dataset generation from rigorous atomistic simulations
4. **Multi-scale architecture evaluation**: Fair comparison of CNNs vs. Transformers
5. **Quantified identifiability limits**: Establishes fundamental boundaries of inverse problem

---

## 9. Knowledge Graph Summary

### Total Nodes: 77
- Research concepts: 15
- Model architectures: 4
- Parameters: 9
- Methods: 8
- Analysis techniques: 3
- Evaluation metrics: 2
- Infrastructure: 1
- Related references: 3

### Total Edges: 95
Relationships include: uses, addresses, implements, evaluates, introduces, contains, enables, varies, governs, isolates, analyzes, specialized_as, applied_to, optimizes

### Key Node Clusters
1. **Hamiltonian Framework**: Extended theory with 8 parameters
2. **Models**: 4 deep architectures with shared regression head
3. **Interpretability**: RAMs with CNN and ViT variants
4. **Magnetic Textures**: 3 critical degenerate features
5. **Analysis**: Phase classification and UMAP
6. **Data Pipeline**: Monte Carlo generation to preprocessing

---

## 10. Knowledge Graph Files Generated

### Primary Output
- **File**: `/papers/Physics_AI/knowledge_graph.json`
- **Format**: JSON with metadata, 77 nodes, 95 edges
- **Size**: Comprehensive extractable structure
- **Structure**:
  - Metadata (paper info, repository links)
  - Research problem definition
  - Methodology overview
  - Nodes array (labeled, typed, described)
  - Edges array (source, target, relation)
  - Key results and insights
  - Code and dataset availability

### Supporting Files
- This summary document (EXTRACTION_SUMMARY.md)

---

## 11. Critical Insights for Future Research

### Physics-Informed ML Success Factors
1. **Rigorous forward modeling**: Monte Carlo ensures physical accuracy
2. **Explicit parameter decomposition**: Clear mapping between interactions and observables
3. **Interpretability coupling**: RAMs bridge latent representations and physical laws
4. **Phase-aware evaluation**: Separate analysis by magnetic regimes reveals true limits

### Identifiability Insights
- **Fundamental limits exist**: J₃, J₄ unidentifiable from 2D s_z projection
- **Not algorithmic failures**: Poor performance in paramagnetic regime is inherent ill-posedness
- **Phase dependence crucial**: R² = 0.97 in ordered phases vs. undefined in disorder

### Architecture Insights
- **Complementary strengths**: ViT captures long-range (R²), Xception captures local (MAE)
- **No universal winner**: Different metrics favor different architectures
- **Hierarchical decomposition**: Network depth correlates with parameter scale

---

## 12. Extracted Concepts and Definitions

**Key Concepts in Knowledge Graph**:
- Inverse problem formulation
- Hamiltonian degeneracy
- Thermal relaxation
- Topological textures (chiral walls, skyrmions, domains)
- Multi-scale feature learning
- Error-decay activation mapping
- Phase space clustering
- Identifiability hierarchy

**Mathematical Foundations**:
- Extended Heisenberg Hamiltonian (9 energy terms)
- Metropolis-Hastings transition probability
- Boltzmann distribution
- Gradient-based attribution
- Multi-head self-attention mechanism
- Depthwise separable convolution

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Deep Learning Models | 4 |
| Hamiltonian Parameters | 9 |
| Identified Parameters | 3 (high identifiability) |
| Degenerate Parameters | 2 (unidentifiable) |
| Magnetic Phases | 4 |
| Critical Textures | 3 |
| Evaluation Metrics | 2 |
| Training Callbacks | 2 |
| Knowledge Graph Nodes | 77 |
| Knowledge Graph Edges | 95 |
| Dataset Samples | 218,256 |
| Splits | 3 (70-15-15) |
| Best R² Score | 0.97 |
| Best MAE Value | 0.039 meV |

