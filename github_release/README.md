# The spatial-scale organization of language-related cortical dynamics

Code, curated numerical outputs, and reproduction materials accompanying *The spatial-scale organization of language-related cortical dynamics*.

## Overview

This study examines how cortical activity during naturalistic language comprehension is organized across spatial scales defined by cortical geometry. Cortical fMRI activity from the English-language *Le Petit Prince* dataset (OpenNeuro **ds003643**) is projected onto Laplace–Beltrami eigenmodes of the `fsaverage5` cortical surface. The geometry-defined basis contains 120 modes per hemisphere; the constant mode is excluded, leaving 119 nonconstant modes. **The eigenmode basis is geometry-defined, whereas the modal-energy spectrum is empirical**, estimated from recorded BOLD activity.

Following nuisance-controlled analyses and stimulus-specific temporal validation, **sentence onset was the only linguistic predictor that survived the corrected cortical validation**. The manuscript's main linguistic analyses therefore focus on sentence onset. Previously examined sentence-shift and token-level effects, and earlier subcortical analyses, are not presented as validated headline findings.

The main analyses address: (i) the empirical distribution of cortical activity across spatial scales; (ii) stimulus-specific validation and vertex-space inference for sentence onset; (iii) its preferential concentration in long-wavelength modes and reconstruction using the first 20 nonconstant eigenmodes; and (iv) geometry-specificity controls using matched spherical modes, leave-one-participant-out (LOPO) functional PCA, folded-versus-spherical subspace overlap, and spatial-null eigenstrapping.

## Reproducing the curated results

From the **`github_release/` directory**, in an environment with Python, NumPy, pandas, SciPy, Matplotlib, scikit-learn, and Jupyter:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_reproduce_paper_figures.ipynb \
  --output 01_reproduce_paper_figures_executed.ipynb \
  --output-dir notebooks --ExecutePreprocessor.timeout=600
```

The notebook checks frozen numerical results against files under `precomputed/` and writes lightweight plots to `reproduced_figures/`. It **does not** recreate the fully designed journal figures, run raw-data preprocessing, re-estimate all GLMs, or rerun expensive permutation tests. The executed notebook is a generated artifact and need not be committed.

The upstream notebook `notebooks/07_vertex_space_linguistic_encoding_FINAL.ipynb` documents the definitive vertex-space, FEM reconstruction, and geometry-specificity analyses; executing it requires intermediate data and mesh resources not packaged in the lightweight release.

## Repository organization

- `notebooks/01_reproduce_paper_figures.ipynb`: canonical lightweight reproduction entry point.
- `notebooks/07_vertex_space_linguistic_encoding_FINAL.ipynb`: upstream computational provenance.
- `precomputed/empirical_spectrum/` and `precomputed/spectral_model_comparison/`: nuisance-corrected empirical modal spectra and model-comparison results.
- `precomputed/sentence_onset/`: controlled temporal validation, vertex inference, FEM reconstruction, and low-mode concentration.
- `precomputed/geometry_specificity/`: spherical-basis comparisons, LOPO PCA, functional overlap, and eigenstrapping spatial-null outputs.
- `precomputed/eigenmodes/`: eigenvalue coordinates used for the empirical spectrum.
- `code/`: selected upstream scripts. The `token_validation/` and `qwen4b/` folders document candidate-effect testing and robustness analyses, **not additional validated effects**.
- `PRECOMPUTED_OUTPUTS.md`: canonical file paths, numerical checks, and provenance qualifications.

## Key numerical results

| Analysis | Result |
|---|---|
| Stimulus-specific sentence-onset modal-energy validation | finite-null p ≈ 0.00010 |
| Bilateral vertex-space group-map RMS | 0.007736873; finite-null p = 0.0359964 |
| FEM-fit/FEM-metric K=20 reconstruction | L R² = 0.608818; R R² = 0.496229 |
| K=20 onset excess concentration over empirical cortical energy | +6.139839 percentage points (49 participants) |
| LOPO PCA vs. folded eigenmodes at K=20 | PCA R² = 0.416575; eigenmodes R² = 0.302087 |
| Folded-minus-spherical functional subspace overlap, K=20 | +0.020570 |
| Folded geometry eigenstrapping spatial null, K=20 | observed 0.556265; null mean 0.521511; finite-null p = 0.00009999 |

PCA's superior reconstruction means that folded eigenmodes should **not** be described as the optimal compression basis. The spatial-null result supports preferential folded-geometry alignment relative to matched spatial surrogates, not uniqueness of the cortical eigenmode representation.

## Preprocessing and inferential provenance

The corrected cortical pipeline removes fMRIPrep nonsteady-state rows; regresses an intercept, six motion parameters and their derivatives, retained aCompCor components, and cosine terms; standardizes vertexwise residuals; and projects into eigenmodes. No global-signal regression was applied. For the sentence-onset temporal null, sentence onset and preceding pause were shifted together on the original full HRF-convolved time grid, while other controlled predictors remained fixed.

The **global vertex-map RMS permutation test** is distinct from **vertexwise maximum-statistic FWER inference**. The FEM reconstruction is likewise distinct from the earlier Euclidean least-squares sensitivity analysis. See `PRECOMPUTED_OUTPUTS.md` for these distinctions and their canonical files.

## Data, citation, and license

The underlying fMRI data are publicly available from OpenNeuro, accession **ds003643**; raw imaging data are not redistributed here. Cite the accompanying manuscript and original dataset when using these outputs. Add the manuscript's bibliographic details and confirm an explicit repository license before public release.
