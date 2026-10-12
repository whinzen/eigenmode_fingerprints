# Canonical precomputed outputs and provenance

This inventory documents the **final nuisance-corrected, sentence-onset-focused** release. Paths are relative to `github_release/`. It supersedes earlier documentation of four purportedly validated linguistic effects and subcortical figure panels.

## Reproduction boundary

`notebooks/01_reproduce_paper_figures.ipynb` checks the curated numerical outputs and generates lightweight diagnostic figures in `reproduced_figures/`. It is not an end-to-end raw-data analysis or a pixel-identical reconstruction of the manuscript artwork. `notebooks/07_vertex_space_linguistic_encoding_FINAL.ipynb` provides upstream computational provenance but requires additional intermediate data and mesh files.

The constant eigenmode is excluded: analyses use **119 nonconstant modes** per hemisphere. The spectrum is **empirical cortical BOLD modal energy**, not an intrinsic energy spectrum.

## Empirical modal-energy spectrum

- `precomputed/empirical_spectrum/energy_spectrum_group_corrected_with_lambda.csv` — corrected group modal energy with eigenvalue coordinate. The coordinate is the canonical **left-hemisphere** eigenvalue sequence from the original spectrum, not an average of left and right eigenvalues.
- `precomputed/empirical_spectrum/energy_spectrum_group_corrected.csv` — corrected group energy without added eigenvalue coordinate.
- `precomputed/empirical_spectrum/energy_spectrum_subject_corrected.csv` — corrected participant spectra.
- `precomputed/eigenmodes/lam_L.npy`, `lam_R.npy` — canonical eigenvalues.
- `precomputed/spectral_model_comparison/group_model_comparison_by_range.csv` — group model-comparison results; further participant and range-sensitivity tables are in the same directory.

Any log-log regression drawn by the reproduction notebook is a visualization; consult the deposited model-comparison tables for formal spectral-model inference.

## Sentence-onset modal validation

- `precomputed/sentence_onset/modal_validation/coherent_sentence_bundle_null_corrected_10000_summary.csv`
- `precomputed/sentence_onset/modal_validation/corrected_empirical_onset_pause_summary.csv`
- `precomputed/sentence_onset/modal_validation/corrected_empirical_onset_pause_beta_star_profiles.csv`

The corrected eight-predictor sentence model includes onset, word rate, content density, preceding pause, sentence shift, acoustic envelope, mean log F0, and pitch reset. The coherent-bundle null jointly circularly shifts onset and preceding pause on the full HRF grid while leaving other covariates fixed. Sentence onset survives corrected stimulus-specific validation (finite-null p ≈ 0.00010). Other candidate predictors do not become validated merely by appearing in upstream scripts or tables.

## Vertex-space onset inference

- `precomputed/sentence_onset/vertex_inference/sentence_onset_beta_star_L.npy`
- `precomputed/sentence_onset/vertex_inference/sentence_onset_beta_star_R.npy`
- `precomputed/sentence_onset/vertex_inference/sentence_onset_vertexwise_inference_summary.csv` — **vertexwise max-stat FWER** results.
- `precomputed/sentence_onset/vertex_inference/sentence_onset_global_RMS_inference.csv` — **distinct bilateral group-map RMS** test.

Definitive bilateral group-map RMS = **0.0077368728815637**, null mean = 0.0058027706754667; 359 of 10,000 null samples equal or exceed the observed statistic, giving finite-null p = **360/10001 = 0.03599640036**. The one-row global RMS CSV was curated from the frozen definitive 10,000-permutation analysis (100 chunks × 100 permutations); it must not be confused with the separate max-stat FWER summary.

## FEM-native K=20 reconstruction

- `precomputed/sentence_onset/reconstruction/sentence_onset_K20_reconstruction_summary.csv` — canonical release summary, where `R2_K20` is the **FEM-fit/FEM-metric** value.
- `precomputed/sentence_onset/reconstruction/sentence_onset_K20_FEM_comparison_summary.csv` — full FEM-versus-Euclidean fitting/evaluation comparison.
- `precomputed/sentence_onset/reconstruction/sentence_onset_K20_reconstruction_L.npy` and `_R.npy` — definitive FEM K20 reconstructions.
- `precomputed/sentence_onset/reconstruction/sentence_onset_empirical_demeaned_L.npy` and `_R.npy` — FEM-demeaned empirical maps.
- `precomputed/sentence_onset/reconstruction/sentence_onset_K20_coefficients_L.npy` and `_R.npy` — FEM K20 coefficients.

The original files were exported by notebook 07, Section 26.2b, then copied with canonical release filenames. FEM-fit/FEM-metric K20 R² = **0.608817683** (L) and **0.496229487** (R). Ordinary Euclidean-fit/Euclidean-metric values are **0.585128873** (L) and **0.472742973** (R); these are sensitivity results, **not** the headline FEM reconstruction. The left/right valid masks contain 9,354/9,361 vertices.

## Preferential long-wavelength concentration

- `precomputed/sentence_onset/concentration/corrected_concentration_inference_K20.csv`
- `precomputed/sentence_onset/concentration/corrected_concentration_inference_all_K.csv`

These tables are restricted to **sentence onset**. The metric compares cumulative absolute linguistic coefficient magnitude with cumulative empirical cortical modal energy. At K20, excess concentration = **6.139839 percentage points**, 95% CI approximately **[4.02, 8.25] percentage points**, N=49. The `mean_delta` field is a fraction; `mean_delta_pp` is its value in percentage points. K20 is the inferential reference; other K cutoffs provide descriptive sensitivity.

## Geometry specificity

- `precomputed/geometry_specificity/cortical_vs_spherical_basis_R2.csv` — matched folded-cortex and spherical eigenmode reconstruction.
- `precomputed/geometry_specificity/sphere_eigengroup_QC.csv` — complete spherical eigengroup QC.
- `precomputed/geometry_specificity/onset_comparison/participant_onset_LB_vs_LOPO_PCA_bilateral.csv` — leave-one-participant-out functional PCA versus folded eigenmodes.
- `precomputed/geometry_specificity/folded_vs_sphere_functional_alignment/group_PCA_folded_vs_sphere.csv` — functional subspace overlap.
- `precomputed/geometry_specificity/spatial_null_eigenstrap/definitive_10000_bilateral_inference.csv`
- `precomputed/geometry_specificity/spatial_null_eigenstrap/definitive_10000_provenance.csv`

At K20, LOPO PCA reconstruction R² = **0.416575**, compared with **0.302087** for folded eigenmodes; PCA is better for 48/49 participants. Folded-minus-spherical functional overlap = **0.020570**. In the population-PCA spherical-eigenstrapping null, observed bilateral overlap = **0.5562653080**, null mean = **0.5215107759**, null SD = **0.0065058636**, and 0/10,000 surrogates exceed the observed value (finite-null p = **1/10001**). The LOPO PCA comparison and population-PCA spatial null are **distinct analyses**; do not conflate their estimands.

## Exclusions and caveats

- Sentence shift, token surprisal, token curvature, token shift, autoregressive residual and subspace exit are **not validated cortical headline effects** in the corrected analysis.
- Earlier hippocampal and VTA analyses are not included as final manuscript claims.
- The folded eigenmode basis is geometry-defined but is **not** the empirically optimal reconstruction basis; functional PCA provides a stronger compression benchmark.
- A significant spatial null does not establish a unique basis or a shared universal mechanism.
- Historical `pang_out/` files are not automatically interchangeable with the curated `precomputed/` files. The reproduction notebook intentionally fails when a required canonical input is absent.
