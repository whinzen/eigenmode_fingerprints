# Eigenmode fingerprints of linguistic structure in cortex

This repository contains code, figure-generation scripts, and reproducibility notebooks for the manuscript:

> **Low-Dimensional Cortical Geometry Organizes Linguistic Representations and Subcortical Updating Signals**

The project analyzes naturalistic fMRI responses to language by projecting cortical activity onto Laplace–Beltrami eigenmodes of the cortical surface.

---

# Overview

The analysis treats cortical activity as a distribution across intrinsic cortical spatial frequencies rather than as localized activation in discrete regions.

For each subject, run, and hemisphere:

1. fsaverage5 cortical BOLD signals are projected onto cortical eigenmodes,
2. mode-wise energy spectra are computed,
3. linguistic regressors are used to estimate how language variables modulate intrinsic cortical spatial modes,
4. reconstructed cortical maps are generated from eigenmode beta profiles.

The manuscript focuses on:

- approximately scale-free, heavy-tailed spatial organization of cortical activity,
- low-dimensional eigenmode structure,
- convergence of sentence-level and token-level linguistic variables onto shared eigenmode profiles,
- reconstruction of cortical maps from low-order eigenmodes,
- hippocampal responses to linguistic updating signals,
- and interactions between cortical, hippocampal, and midbrain systems during naturalistic language comprehension.

# Top-level architecture diagram

Stimulus transcripts
        ↓
 Linguistic regressors
        ↓
 Cortical eigenmodes ──────→ cortical GLMs
        ↓
 Hippocampal eigenmodes ───→ hippocampal GLMs
        ↓
      VTA / LC
        ↓
 Coupling analyses

---

# Repository structure

Expected layout:

```text
eigenmode_fingerprints/
├── README.md
├── run_pipeline.sh
├── code/
├── notebooks/
│   └── 01_reproduce_paper_figures.ipynb
├── modes/fsaverage5/
│   ├── L_phi.npy
│   ├── R_phi.npy
│   ├── L_lam.npy
│   └── R_lam.npy
├── data/empirical/sub-*/func/
│   └── *_hemi-*_space-fsaverage5_bold.func.gii
├── ds003643/annotation/EN/repunct/
│   ├── lppEN.csv
│   └── little_prince_series_curvature.csv
└── pang_out/
```

The repository assumes:

- preprocessed fsaverage5 surface BOLD data,
- precomputed cortical eigenmodes,
- and precomputed group-level outputs.

Large raw neuroimaging files are not included in the repository.

---

# Environment

Main Python dependencies:

- numpy
- pandas
- scipy
- matplotlib
- nibabel
- nilearn
- neuromaps

Optional:

- jupyter
- notebook

---

# Connectome Workbench

Some neuromaps comparisons require Connectome Workbench.

Example setup:

```bash
export PATH=~/tools/workbench/bin_linux64:$PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Check installation:

```bash
wb_command -version
```

---

# Reproducing manuscript figures

The main reproducibility workflow is:

```text
notebooks/01_reproduce_paper_figures.ipynb
```

This notebook regenerates manuscript figures and tables from precomputed outputs stored in:

```text
pang_out/
```

The notebook does **not** rerun the full fMRI preprocessing or eigenmode projection pipeline.

---

# Required precomputed outputs

The following folders are required to reproduce the figures:

```text
modes/fsaverage5/
pang_out/word_transition_geometry/
pang_out/vertex_betas/
pang_out/group_sentence_level_glm/
pang_out/group_shift_glm/
pang_out/group_pred_error_ar_glm/
pang_out/group_pred_error_subspace_glm/
pang_out/group_curvature_glm/
pang_out/group_boundary_wordrate_content_glm/
pang_out/standardized_beta_profiles/
pang_out/mode3_annotations/
pang_out/paper_tables/
```

Optional ready-made figure outputs:

```text
pang_out/paper_figures/
```

---

# Running figure generation

Run:

```bash
bash run_pipeline.sh
```

Outputs are written to:

```text
pang_out/paper_figures/
pang_out/paper_tables/
```

---

# Full analysis pipeline

The complete workflow consists of the following stages.

---

## 1. Eigenmode energy spectra

Scripts:

```text
code/energy/energy_compute_per_run.py
code/energy/energy_all_collect.py
code/energy/energy_group_aggregate.py
code/energy/energy_fit_criticality.py
code/energy/energy_fit_group_criticality.py
code/energy/energy_criticality_diagnostics.py
code/energy/make_energy_table.py
```

These scripts compute:

- eigenmode coefficients,
- mode-wise energy spectra,
- group-level spectral summaries,
- and scale-free diagnostics.

---

## 2. Sentence-level regressors

Scripts:

```text
build_sentence_boundary_regressors_per_subject.py
build_sentence_shift_regressors.py
rebuild_glm_boundary_per_run.py
rebuild_glm_sentence_shift_per_run.py
group_aggregate_sentence_level.py
```

These generate:

- sentence-boundary regressors,
- sentence-level representational-shift regressors,
- eigenmode GLMs,
- and group-level beta profiles.

---

## 3. Token-level transition metrics

Scripts:

```text
analyze_word_transition_geometry.py
build_transition_metric_regressors.py
rebuild_glm_transition_metric_per_run.py
group_aggregate_transition_metric.py
```

These estimate token-level metrics including:

- representational shift,
- autoregressive prediction error,
- subspace-exit error,
- and curvature.

---

## 4. Curvature-family analyses

Scripts:

```text
build_curvature_regressors_per_subject.py
rebuild_glm_curvature_per_run.py
group_aggregate_curvature.py
figure_curvature_family_collapse.py
```

These analyze multiple operationalizations of embedding-space trajectory geometry.

---

## 5. Word-rate control analyses

Scripts:

```text
build_wordrate_regressors_per_subject.py
rebuild_glm_wordrate_per_run.py
group_aggregate_wordrate.py
rebuild_glm_joint_wordrate_per_run.py
group_aggregate_joint_wordrate.py
rebuild_glm_joint_wordrate_residualized_per_run.py
group_aggregate_joint_wordrate_residualized.py
```

These generate TR-aligned word-rate regressors and estimate both joint and residualized control models.

---

## 6. Cortical spectral-envelope control

Because modal energy differs substantially across cortical eigenmodes, we tested
whether the high similarity of linguistic beta profiles could arise trivially
from mode-dependent differences in the scale or variance of the modal-energy
signal.

Script:

```text
code/standardized_eigenmode_profiles.py
```

For each run, hemisphere, predictor, and eigenmode, the standardized effect is

```text
beta_std = beta * SD(X) / SD(E)
```

where `X` is the HRF-convolved linguistic predictor and `E = A^2` is the
modal-energy time series. For the single-predictor models used here, this is
equivalent to the Pearson correlation between the predictor and modal energy:

```text
beta_std = r(X, E)
```

This standardization removes mode-dependent differences in modal-energy
variance. The resulting token-level profiles remained nearly identical across
linguistic predictors (pairwise r = 0.9994–0.9998), indicating that their
convergence cannot be explained by inheritance of the generic cortical
modal-energy envelope.

Outputs:

```text
pang_out/standardized_beta_profiles/
    standardized_effects_allruns_token.csv
    standardized_effects_subject_token.csv
    standardized_effects_group_token.csv
    standardized_group_profiles_token.csv
    profile_correlations_token.csv
```

---

## 7. Spatial-scaling robustness and null models

The robustness analyses underlying the revised Figure 2 are provided in:

    notebooks/03_spatial_scaling_null_models.ipynb

This notebook reproduces the empirical cortical eigenmode spectrum and tests
whether its approximately scale-free, heavy-tailed form can be explained by
generic spatial smoothness. It includes the primary power-law fit over modes
1–60, an analytical heat-kernel spatial-smoothing null, a parametric-bootstrap
goodness-of-fit test, explicit heat diffusion of random maps on the fsaverage5
cortical mesh, comparison of power-law, exponential, stretched-exponential,
and power-law-with-cutoff spectral models, participant-level model
comparisons, and robustness analyses across alternative eigenmode fitting
ranges.

The analytical and explicit mesh-based nulls reproduce the broad low-frequency
dominance expected from spatial smoothness, but do not reproduce the empirical
spectral form. The analyses therefore test the stronger claim that the observed
spectrum is not explained by generic cortical smoothness alone.

---
## 8. Manuscript figures and tables

Representative scripts:

```text
plot_linguistic_composite_AB.py
figure_sentence_level_empirical_reconstruction_twopanel.py
figure_modes_plus_beta_pial.py
figure_empirical_full_lowpass_residual.py
analyze_reconstructed_map_correlations.py
make_empirical_reconstruction_summary_table.py
compare_mode3_to_gradient1_template.py
```

Outputs are written to:

```text
pang_out/paper_figures/
pang_out/paper_tables/
```

## 9. Subcortical analyses

The repository additionally contains a complete subcortical analysis framework located in:

```text
code/subcortex/
```

These analyses extend the cortical eigenmode framework to hippocampal and brainstem systems involved in memory updating and predictive processing during naturalistic language comprehension.

The reproducibility notebook includes both the cortical analyses reported in the main manuscript and the hippocampal/VTA analyses reported in the subcortical extension of the framework.

### Hippocampal analyses

Representative scripts:

```text
code/subcortex/subcortex_hippocampus_compute_all.py
code/subcortex/rebuild_glm_hipp_sentence_level.py
code/subcortex/rebuild_glm_hipp_token_level.py
code/subcortex/rebuild_glm_hipp_mean_signal.py
code/subcortex/rebuild_glm_hipp_mean_signal_AP.py
code/subcortex/hipp_compute_trajectory_features.py
code/subcortex/rebuild_hipp_trajectory_glm_with_mean_signal_covariate.py
```

These analyses include:

- hippocampal graph-eigenmode decomposition,
- hippocampal energy-spectrum estimation,
- sentence-level and token-level GLMs,
- anterior–posterior hippocampal segmentation,
- trajectory analyses in hippocampal eigenmode space,
- and boundary-related hippocampal dynamics.

### Brainstem analyses

Representative scripts:

```text
code/subcortex/extract_brainstem_roi_timeseries.py
code/subcortex/rebuild_glm_brainstem_roi.py
code/subcortex/rebuild_glm_vta_multivariate.py
code/subcortex/rebuild_glm_vta_to_cortical_energy.py
code/subcortex/rebuild_glm_vta_to_cortical_energy_residualized.py
```

These analyses include:

- extraction of VTA and LC ROI time series,
- linguistic GLMs within brainstem nuclei,
- multivariate VTA analyses,
- and VTA–cortical coupling in cortical eigenmode space.

### Coupling analyses

Representative scripts:

```text
code/subcortex/compute_vta_hipp_timeseries_coupling.py
code/subcortex/compute_vta_hipp_coupling_global_residualized.py
code/subcortex/compute_boundary_modulated_vta_hipp_coupling.py
code/subcortex/compare_boundary_and_vta_cortical_eigenmodes.py
```

These analyses quantify:

- VTA–hippocampal coupling,
- global-signal-controlled coupling,
- sentence-boundary modulation of coupling,
- and overlap between cortical sentence-boundary effects and VTA-driven cortical eigenmode profiles.

### Subcortical figure generation

Representative figure scripts:

```text
subcortex/plot_hippocampus_all_results_summary.py
subcortex/plot_main_subcortical_updating_network.py
subcortex/plot_main_vta_cortical_overlap.py
subcortex/plot_vta_hipp_coupling_global_residualized.py
subcortex/plot_vta_hipp_predictor_betas.py
```

---

# Notes

## Eigenmode indexing

Mode 0 corresponds to the near-constant spatial mode and is excluded from most interpretive analyses.

Reported effects focus on non-constant low-order modes, particularly Mode 3.

---

## Projection into eigenmode space

For each time point:

```math
A_k(t) = \langle X_t, \phi_k \rangle
```

where:

- $begin:math:text$X\_t$end:math:text$ is the cortical BOLD pattern,
- $begin:math:text$\\phi\_k$end:math:text$ is eigenmode $begin:math:text$k$end:math:text$,
- and $begin:math:text$A\_k\(t\)$end:math:text$ is the corresponding mode amplitude.

The linguistic regressors themselves are not spatially projected. Eigenmode decomposition is applied only to cortical activity.

---

## Reconstructed cortical maps

Reconstructed maps are generated from eigenmode beta profiles:

```math
\hat{m}(v) = \sum_{k=1}^{K} \beta_k \phi_k(v)
```

where:

- $begin:math:text$\\phi\_k\(v\)$end:math:text$ is eigenmode $begin:math:text$k$end:math:text$ at cortical vertex $begin:math:text$v$end:math:text$,
- and $begin:math:text$\\beta\_k$end:math:text$ is the GLM beta weight for mode $begin:math:text$k$end:math:text$.

Low-pass reconstructions retain only the first $begin:math:text$K$end:math:text$ modes.

---

## Interpretation of low-order dominance

Low-order eigenmodes naturally capture much of large-scale cortical activity,
and generic spatial smoothness can reproduce broad low-frequency dominance.
However, smooth-field and analytical heat-kernel null models do not reproduce
the empirical spectral form. Across the observed eigenmode range, the cortical
spectrum is substantially better described by approximately scale-free or
heavy-tailed organization than by simple exponential spectral decay.

Likewise, the convergence of linguistic beta profiles is not a trivial
consequence of the non-uniform cortical modal-energy spectrum. Token-level
profiles remain nearly identical after expressing effects as standardized
coefficients that remove mode-dependent differences in modal-energy variance.

These analyses distinguish the generic spectral organization of cortical
activity from the shared mode-specific cortical response profile associated
with linguistic variables. Whether this latter profile is specific to
language or instead reflects a more general cortical event-response geometry
remains an open question.

---

# Reproducibility workflow

## Figures only

Open:

```text
notebooks/01_reproduce_paper_figures.ipynb
```

---

## Full figure regeneration

Run:

```bash
bash run_pipeline.sh
```

---

## Full raw-data re-execution

The complete pipeline additionally requires:

- the LPP-EN dataset,
- fsaverage5 surface BOLD files,
- and subject-level preprocessing outputs,

which are not distributed with this repository.
