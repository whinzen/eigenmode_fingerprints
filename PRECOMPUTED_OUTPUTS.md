# Canonical precomputed outputs and manuscript dependency map

This document identifies the **canonical precomputed inputs** used to reproduce the main manuscript figures in *The scale-spanning cortical geometry of language*.

The repository's `pang_out/` tree is a historical analysis-output tree. It contains both final manuscript results and superseded exploratory analyses. The existence of a file in `pang_out/` therefore does **not** imply that it contributes to the final manuscript. The paths below define the manuscript-facing subset.

## Scope

- Figure 1 is conceptual/schematic and is not a numerical reproduction target.
- Figures 2–6 are reproduced from the precomputed numerical outputs listed below.
- The reproduction notebook should use these explicit paths rather than searching `pang_out/` for the first matching filename.
- The global/constant cortical eigenmode is excluded; cortical analyses use 119 nonconstant modes.
- The Figure 2 reference distribution is the **empirical cortical modal-energy spectrum during narrative listening**, not an “intrinsic cortical energy” spectrum. “Intrinsic” is reserved for the geometry-defined eigenmode basis.

## Figure 2 — Empirical cortical modal-energy spectrum and spatial-scaling controls

**Definitive source notebook:** `03_spatial_scaling_null_models(1).ipynb`

Primary empirical spectrum:

```text
pang_out/group/energy_spectrum_group.csv
pang_out/group/energy_spectrum_subject.csv
```

`energy_spectrum_group.csv` supplies `mode_k`, eigenvalue `lam`, group mean modal energy `Emean`, and `Esem`. The primary log-log fit is over retained modes 1–60. `energy_spectrum_subject.csv` supports participant-level validation and robustness analyses.

Spatial-smoothing nulls:

```text
pang_out/spatial_smoothing_null/expected_smoothing_spectra_grid.csv
pang_out/mesh_spatial_smoothing_null/mesh_smoothing_spectra.csv
pang_out/mesh_spatial_smoothing_null/mesh_smoothing_comparison.csv
```

Spectral-model comparison and fitting-range robustness:

```text
pang_out/spectral_model_comparison/group_model_comparison_by_range.csv
pang_out/spectral_model_comparison/participant_model_comparison_by_range.csv
pang_out/spectral_model_comparison/powerlaw_vs_exponential_range_robustness.csv
pang_out/spectral_model_comparison/participant_winner_counts_by_range.csv
```

Canonical outputs:

```text
pang_out/paper_figures/spatial_scaling_revision/figure2_revised_spatial_scaling.pdf
pang_out/paper_figures/spatial_scaling_revision/figure2_revised_spatial_scaling.png
pang_out/paper_figures/spatial_scaling_revision/supplementary_spatial_scaling_robustness.pdf
pang_out/paper_figures/spatial_scaling_revision/supplementary_spatial_scaling_robustness.png
pang_out/paper_figures/spatial_scaling_revision/supplementary_table_1_candidate_models.csv
pang_out/paper_figures/spatial_scaling_revision/supplementary_table_1_candidate_models_formatted.csv
pang_out/paper_figures/spatial_scaling_revision/supplementary_table_2_range_robustness.csv
pang_out/paper_figures/spatial_scaling_revision/supplementary_table_2_range_robustness_formatted.csv
```

**Status:** the numerical inputs and final publication outputs are present in the supplied `pang_out` listing.

## Figure 3 — Stimulus-specific validation and preferential long-wavelength concentration

**Definitive cortical source notebook:** `04_qwen_continuous_context_robustness_CLEAN_20260922(3).ipynb`

Figure 3 combines stimulus-specific temporal validation with raw partial-beta allocation across eigenmodes for the four effects that survived validation.

Sentence-level controlled model and participant table:

```text
pang_out/group_sentence_controls_exact_8pred/
pang_out/group_sentence_controls_exact_8pred/sentence_controls_8pred_standardized_subject_level.csv
```

Sentence-onset coherent-bundle null:

```text
pang_out/group_sentence_controls_exact_8pred/coherent_sentence_bundle_null_8pred_test/coherent_sentence_bundle_null_8pred_10000.csv
pang_out/group_sentence_controls_exact_8pred/coherent_sentence_bundle_null_8pred_test/coherent_sentence_bundle_null_8pred_10000_summary.csv
```

Sentence-shift null:

```text
pang_out/group_sentence_controls_exact_8pred/sentence_shift_shuffle_null/sentence_shift_shuffle_null_1000.csv
pang_out/group_sentence_controls_exact_8pred/sentence_shift_shuffle_null/sentence_shift_shuffle_null_10000_summary.csv
```

The saved 1,000-permutation shift distribution is used for display; the 10,000-permutation summary is the definitive inferential result.

Token-level upstream-of-HRF temporal validation:

```text
pang_out/token_upstream_hrf_temporal_null_test/token_upstream_hrf_null_10000.csv
pang_out/token_upstream_hrf_temporal_null_test/token_upstream_hrf_null_10000_summary.csv
```

Only **Qwen surprisal** and **curvature** survive this validation and enter the final four-effect cortical geometry.

Token surprisal and curvature final-analysis roots:

```text
pang_out/standardized_beta_profiles_qwen3_0p6b_continuous_centered_wordrate_surprisal/
pang_out/group_qwen3_0p6b_continuous_centered_wordrate_curvature_glm/
```

Empirical cortical modal-energy reference:

```text
pang_out/group/energy_spectrum_group.csv
pang_out/group/energy_spectrum_subject.csv
```

Final derived concentration results:

```text
pang_out/validated_linguistic_effects_longwavelength/
```

Canonical outputs:

```text
pang_out/paper_figures/figure3_validation_and_longwavelength_concentration.pdf
pang_out/paper_figures/figure3_validation_and_longwavelength_concentration.png
```

**Important:** old outputs for token shift, autoregressive trajectory residual, and subspace exit remain in `pang_out/`; they are historical analyses and are not final Figure 3 effects.

## Figure 4 — Low-dimensional geometry of the four validated effects

**Definitive cortical source notebook:** `04_qwen_continuous_context_robustness_CLEAN_20260922(3).ipynb`

The final analysis uses **variance-standardized partial beta-star profiles** from the fully controlled models. It does not use the raw-beta profiles from Figure 3.

Final controlled sentence profiles:

```text
pang_out/group_sentence_controls_exact_8pred/sentence_8pred_hemiavg_profiles.csv
```

Final controlled token profiles:

```text
pang_out/standardized_beta_profiles_qwen3_0p6b_continuous_centered_wordrate/standardized_group_profiles_token.csv
pang_out/standardized_beta_profiles_qwen3_0p6b_continuous_centered_wordrate_surprisal/standardized_group_profile_surprisal.csv
```

Canonical four-effect geometry outputs:

```text
pang_out/validated_linguistic_effects_geometry/validated_four_beta_star_profiles.csv
pang_out/validated_linguistic_effects_geometry/validated_four_shapePCA_component_weights.csv
pang_out/validated_linguistic_effects_geometry/validated_four_shapePCA_explained_variance.csv
```

Canonical outputs:

```text
pang_out/paper_figures/figure4_validated_effects_lowdimensional_geometry.pdf
pang_out/paper_figures/figure4_validated_effects_lowdimensional_geometry.png
```

PCA signs are arbitrary; the source analysis orients PC1 so sentence onset is positive and PC2 so sentence shift is positive.

## Figure 5 — K=20 cortical reconstruction of validated effects

**Definitive cortical source notebook:** `04_qwen_continuous_context_robustness_CLEAN_20260922(3).ipynb`

The final figure uses only:

```text
sentence_onset
sentence_shift
token_surprisal
token_curvature
```

Canonical reconstruction root:

```text
pang_out/validated_effects_reconstruction/
```

For each effect and hemisphere (`L`, `R`), the final figure loads:

```text
<effect>/<effect>_hemi-<H>_empirical.npy
<effect>/<effect>_hemi-<H>_low.npy
<effect>/<effect>_hemi-<H>_residual.npy
```

Reconstruction provenance additionally includes:

```text
<effect>/<effect>_hemi-<H>_full.npy
<effect>/<effect>_hemi-<H>_coef_full.npy
<effect>/<effect>_hemi-<H>_coef_K20.npy
```

Reconstruction statistics:

```text
pang_out/validated_effects_reconstruction/sentence_reconstruction_diagnostics.csv
pang_out/validated_effects_reconstruction/token_reconstruction_diagnostics.csv
pang_out/validated_effects_reconstruction/validated_four_effect_reconstruction_diagnostics.csv
```

Upstream reconstruction inputs:

```text
modes/fsaverage5/
pang_out/vertexwise_validated_effects/sentence_8pred/
pang_out/vertexwise_validated_effects/token_wordrate_controlled/
```

These are provenance for rebuilding the saved reconstruction arrays; the manuscript-level reproduction notebook need not recompute them.

Canonical outputs:

```text
pang_out/paper_figures/figure5_validated_effects_K20_reconstruction_FINAL.pdf
pang_out/paper_figures/figure5_validated_effects_K20_reconstruction_FINAL.png
```

**Important:** older six-effect reconstruction files for `token_shift`, `pred_error_ar`, and `pred_error_subspace` are not final Figure 5 dependencies.

## Figure 6 — Subcortical and cortico-subcortical coordination

**Definitive source notebook:** `05_subcortex_validated_effects_CLEAN_20260922(3).ipynb`

The final Figure 6 cell explicitly reloads participant-level outputs and **does not recompute the GLMs**.

Canonical root:

```text
pang_out/subcortex/validated_effects_20260921/
```

Panel A — hippocampal mean-signal responses:

```text
hippocampal_mean_signal/hipp_mean_validated_effects_participant_bilateral.csv
```

Panel B — hippocampal eigenspace responses:

```text
hippocampal_eigenspace_language/hc_eigenspace_bilateral_participant.csv
```

Panel C — VTA mean-signal responses:

```text
vta_mean_signal/vta_bilateral_participant.csv
```

Panel D — baseline VTA–hippocampal coupling:

```text
vta_hipp_coupling/vta_hc_coupling_subject_bilateral.csv
```

Panel E — sentence modulation of VTA–hippocampal cofluctuation:

```text
vta_hipp_sentence_cofluctuation/vta_hc_cofluctuation_bilateral_participant.csv
```

Panel F — sentence modulation of VTA–cortical modal cofluctuation:

```text
vta_cortical_sentence_cofluctuation/vta_cortical_sentence_cofluctuation_participant_bilateral.csv
vta_cortical_sentence_cofluctuation/sentence_modulation_LOO_paired_participants.csv
vta_cortical_sentence_cofluctuation/sentence_modulation_LOO_inference.csv
```

Canonical outputs:

```text
pang_out/subcortex/validated_effects_20260921/figures/Figure6_subcortical_validated_effects.pdf
pang_out/subcortex/validated_effects_20260921/figures/Figure6_subcortical_validated_effects.png
pang_out/subcortex/validated_effects_20260921/figures/Figure6_subcortical_validated_effects.svg
```

## Manuscript-level dependency graph

```text
Figure 2
  group modal-energy spectrum
      + analytical smoothing null
      + explicit mesh smoothing null
      + spectral-model comparison
      -> Figure 2 + spatial-scaling supplement

Figure 3
  final 8-predictor sentence GLM
      + sentence temporal nulls
  continuous-context Qwen token GLMs
      + upstream-of-HRF token temporal null
  participant cortical modal energy
      -> validated four effects
      -> raw-beta profiles
      -> cumulative long-wavelength concentration
      -> Figure 3

Figure 4
  final controlled beta-star profiles
      -> retain four validated effects
      -> profile correlations
      -> shape PCA
      -> Figure 4

Figure 5
  controlled vertexwise partial-beta maps
      + fsaverage5 cortical eigenmodes
      -> full and K=20 reconstructions
      -> saved empirical / K20 / residual maps
      -> Figure 5

Figure 6
  validated participant-level hippocampal and VTA outputs
      + VTA–hippocampal coupling/cofluctuation
      + VTA–cortical modal cofluctuation
      -> Figure 6
```

## Historical outputs that are not final manuscript dependencies

The supplied `pang_out` listing contains many valid analyses from earlier stages. The final manuscript notebook should not automatically ingest:

- six-effect reconstructions containing `token_shift`, `pred_error_ar`, or `pred_error_subspace`;
- BERT-versus-Qwen development figures;
- old token-transition figures;
- exploratory residualized-profile analyses;
- curvature-family diagnostics;
- old group beta-profile PCA outputs;
- old sentence-only reconstruction figures;
- files under `pang_out/OLD/`;
- Apple resource-fork files beginning `._`.

They may be retained for provenance but are not canonical inputs for Figures 2–6.

## Recommended reproduction policy

The manuscript-facing notebook should be deliberately **thin**:

1. validate that every canonical input listed above exists;
2. load the stored numerical outputs;
3. regenerate Figures 2–6 and headline numerical summaries;
4. write figures to one canonical output directory;
5. fail with a clear `FileNotFoundError` if a required input is absent.

It should **not** recursively search `pang_out/` for plausible alternatives and should not rerun expensive raw-data preprocessing. Upstream notebooks/scripts remain the provenance record for producing the precomputed outputs.

This supports the precise public claim:

> The repository provides the upstream analysis code and a manuscript-level reproduction notebook that regenerates the main figures and key numerical results from precomputed analysis outputs.

It does **not** imply that a single shell command reprocesses the raw OpenNeuro data through every upstream stage.
