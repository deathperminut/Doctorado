# State of the Art - Problem Statements

## Overview

This directory contains detailed analyses of the main research problems and challenges in magnetic domain characterization using deep learning and Bayesian methods.

## Structure

### ✅ ProblemStatement1 - Non-Uniqueness in Magnetic Domains.md (39KB)
**Status**: Complete

**Topics Covered**:
- Mathematical formulation of the inverse problem
- Physical origin of parameter degeneracy
- Forward approaches (Differentiable simulators, PINNs, Generative models)
- Inverse approaches (Bayesian inference, Cycle-consistent, Hybrid methods)
- Comparative analysis with decision tree
- State-of-the-art works (Ahmad 2023, Müller 2024, ICLR 2025, etc.)
- Research gaps and future directions
- Concrete recommendations for PhD research

**Key Sections**:
1. Non-Uniqueness Problem (math + physics)
2. Forward Approaches (4 methods)
3. Inverse Approaches (4 methods)
4. Comparative Analysis (tables + metrics)
5. SOTA Works (4 major papers)
6. Research Gaps
7. Recommendations (short/medium/long term)
8. Code Repository Structure

### ✅ ProblemStatement2 - Low Quality Data Mitigation.md (50KB)
**Status**: Complete

**Topics Covered**:
- Data quality issues (resolution, noise, artifacts, sim-to-real gap)
- Forward mitigation (instrument-aware modeling, tip deconvolution)
- Inverse mitigation (denoising, super-resolution, domain adaptation)
- Physics-informed reconstruction
- Generative priors (diffusion models, GANs)
- Comparative analysis with decision matrix
- State-of-the-art works (PMC 2024, PubMed 2021, MDPI 2022/2024)
- Research roadmap (short/medium/long term)

**Key Sections**:
1. Low-Quality Data Problem (sources + impact)
2. Forward Strategies (3 methods)
3. Inverse Strategies (4 methods)
4. Comparative Analysis (benchmarks + decision tree)
5. SOTA Works (4 major papers)
6. Recommended Pipeline
7. Implementation Resources

### ✅ ProblemStatement3 - Interpretability and Physical Consistency.md (75KB)
**Status**: Complete

**Topics Covered**:
- Interpretability problem in physics (black-box issues, failure modes)
- Post-hoc methods (Saliency, Grad-CAM, SHAP, LIME)
- Physics-constrained architectures (HNNs, Equivariant NNs, Hard constraints)
- Symbolic regression and equation discovery (PySR, SINDy, distillation)
- Mechanistic interpretability (Probing, circuit analysis, concepts)
- Physical consistency verification (Conservation, symmetry, bounds testing)
- Comparative analysis with decision tree
- State-of-the-art works (Cranmer 2023, Batzner 2022, Lample 2024, Jiang 2024)
- Research roadmap (short/medium/long term)

**Key Sections**:
1. Interpretability Problem (3 failure modes)
2. Post-Hoc Methods (4 techniques)
3. Physics-Constrained Architectures (3 approaches)
4. Symbolic Regression (3 methods)
5. Mechanistic Interpretability (2 approaches)
6. Physical Consistency Verification (3 tests)
7. Comparative Analysis (decision tree + table)
8. SOTA Works (8 major papers from 2022-2024)
9. Research Roadmap
10. Implementation Resources

## Related Documents

**In `literatura/` directory**:
- `Bayesian Optimization.md` (67KB) - Comprehensive guide to hyperparameter optimization
- `Bayesian Physics-Informed Neural Networks.md` (90KB+) - Detailed B-PINNs methodology

**Connection**:
- Problem Statement 1 references both Bayesian Optimization and B-PINNs
- Provides context for when to use each technique
- Integrates methods into unified inverse problem framework
- Problem Statement 2 builds on PS1 by addressing data quality challenges
- Problem Statement 3 ensures models are interpretable and physically consistent
- All three form complete pipeline: Inverse method + Data handling + Validation

## Usage

### For Literature Review
Read in order:
1. Bayesian Optimization (foundation)
2. B-PINNs (physics-informed methods)
3. ProblemStatement1 (non-uniqueness + inverse methods)
4. ProblemStatement2 (data quality + mitigation strategies)
5. ProblemStatement3 (interpretability + physical consistency)

### For Research Planning
- Use ProblemStatement1 Section 8 (Recommendations) for inverse problem approaches
- Use ProblemStatement2 Section 6 (Recommended Pipeline) for data quality mitigation
- Use ProblemStatement3 Section 9 (Research Roadmap) for interpretability integration
- Follow short/medium/long term milestones across all three documents
- Reference comparative analysis sections for method selection
- Combine strategies: Physics-constrained interpretable models with data quality mitigation

### For Implementation
- ProblemStatement1 Section 10: Code repository for inverse methods
- ProblemStatement2 Section 7: Code repository for data mitigation
- ProblemStatement3 Section 10: Code repository for interpretability
- 48+ complete code blocks across all documents
- Links to open-source implementations:
  - Inverse methods: Pyro, Spirit, mumax3
  - Data mitigation: DenoiSeg, ESRGAN, Transfer Learning Library
  - Interpretability: Captum, e3nn, escnn, PySR, PySINDy

## Spanish Translations

All three Problem Statement documents are now available in Spanish:

### ✅ ProblemStatement1 - Non-Uniqueness in Magnetic Domains_ES.md (1135 lines)
**Full Spanish translation** of Problem Statement 1
- Main title kept in English
- All content, tables, and references translated to Spanish
- All code blocks and LaTeX equations preserved unchanged

### ✅ ProblemStatement2 - Low Quality Data Mitigation_ES.md (1565 lines)
**Full Spanish translation** of Problem Statement 2
- Main title kept in English
- Complete translation including all 15+ code implementations
- All comparative tables and SOTA works with URLs translated

### ✅ ProblemStatement3 - Interpretability and Physical Consistency_ES.md (2158 lines)
**Full Spanish translation** of Problem Statement 3
- Main title kept in English
- Comprehensive translation including all 25+ code blocks
- Complete interpretability methods and research roadmap in Spanish

**Note**: Spanish versions maintain identical structure and technical content as English originals. Only descriptive text is translated; code, equations, and URLs remain unchanged.

## Statistics

| Document | Size | Lines | Sections | Code Blocks | References | Status |
|----------|------|-------|----------|-------------|------------|--------|
| ProblemStatement1 (EN) | 39KB | 1135 | 10 | 8 | 9 papers | ✅ Complete |
| ProblemStatement1 (ES) | 39KB | 1135 | 10 | 8 | 9 papers | ✅ Complete |
| ProblemStatement2 (EN) | 50KB | 1565 | 9 | 15+ | 10 papers | ✅ Complete |
| ProblemStatement2 (ES) | 50KB | 1565 | 9 | 15+ | 10 papers | ✅ Complete |
| ProblemStatement3 (EN) | 75KB | 2158 | 12 | 25+ | 15 papers | ✅ Complete |
| ProblemStatement3 (ES) | 75KB | 2158 | 12 | 25+ | 15 papers | ✅ Complete |

## Next Steps

1. **Review ProblemStatement1**:
   - Add figures (4 TODOs marked)
   - Validate code examples
   - Cross-check references

2. **Review ProblemStatement2**:
   - Add figures (2 TODOs marked)
   - Test denoising implementations
   - Validate domain adaptation code

3. **Review ProblemStatement3**:
   - Add figures (1 TODO marked - symmetry violation example)
   - Test interpretability implementations (Grad-CAM, SHAP, probing)
   - Validate equivariant architectures (e3nn, escnn)
   - Run symbolic regression examples

4. **Integration**:
   - Cross-reference concepts across all three documents
   - Create unified implementation roadmap
   - Identify synergies (e.g., physics-constrained denoising)

---

**Last Updated**: December 8, 2025
**Maintainer**: Juan Sebastián Méndez Rondón
**Project**: PhD Thesis - Magnetic Domain Characterization
