# Canonical precomputed outputs and analysis provenance

This document describes the curated numerical inputs used to reproduce the manuscript-level results of **The scale-spanning cortical geometry of language**.

The canonical reproduction notebook is:

`notebooks/01_reproduce_paper_figures.ipynb`

The tested command-line entry point is:

```bash
bash reproduce_figures.sh
```

## 1. Scope

The final manuscript-facing analyses concern:

1. The empirical distribution of cortical BOLD energy across Laplace–Beltrami eigenmodes.
2. Validation and spatial organization of sentence-onset responses.
3. Preferential concentration of sentence-onset responses in long-wavelength cortical modes.
4. Specificity of functional organization to folded cortical geometry, including spatial-null inference.

Earlier analyses involving multiple linguistic predictors and subcortical responses remain available for provenance but are not part of the final manuscript reproduction workflow.

The reproduction notebook uses curated numerical outputs rather than recomputing the underlying analyses from raw neuroimaging data.

## 2. Canonical input directory

All paths below are relative to:

```text
github_release/precomputed/
```

The canonical notebook explicitly checks that all 20 required files exist before proceeding.

### A. Empirical cortical energy spectrum

```text
empirical_spectrum/energy_spectrum_group_corrected_with_lambda.csv
empirical_spectrum/energy_spectrum_subject_corrected.csv
spectral_model_comparison/group_model_comparison_by_range.csv
```

The group spectrum contains modal indices, Laplace–Beltrami eigenvalues, modal energies, and participant-level variability summaries.

The eigenvalue-containing version is required for the log-log spectral fit. The reproduction notebook fits modes 1–60 and excludes the approximately constant mode 0.

### B. Sentence-onset validation

```text
sentence_onset/modal_validation/coherent_sentence_bundle_null_corrected_10000_summary.csv
sentence_onset/modal_validation/corrected_empirical_onset_pause_summary.csv
sentence_onset/modal_validation/corrected_empirical_onset_pause_beta_star_profiles.csv
```

These files contain the empirical sentence-onset effects and the results of stimulus-specific temporal validation.

The structured temporal null preserves the organization of sentence-boundary events through coherent bundle shifts.

### C. Vertex-space inference

```text
sentence_onset/vertex_inference/sentence_onset_vertexwise_inference_summary.csv
sentence_onset/vertex_inference/sentence_onset_global_RMS_inference.csv
sentence_onset/vertex_inference/sentence_onset_beta_star_L.npy
sentence_onset/vertex_inference/sentence_onset_beta_star_R.npy
```

Two distinct inferential analyses are represented.

**Vertexwise FWER inference** evaluates significance across individual cortical vertices, correcting for multiple comparisons.

**Global RMS inference** evaluates the overall magnitude of the bilateral sentence-onset response against a structured temporal null.

The definitive global RMS inference reports:

- Observed bilateral RMS: 0.0077368728815637
- Null mean: 0.0058027706754667
- Number of permutations: 10,000
- Null values greater than or equal to observed: 359
- One-sided finite-surrogate p-value: 0.035996400359964

These statistics should not be confused with the minimum vertexwise FWER-corrected p-values.

The RMS values reported in the vertexwise summary differ slightly from those computed from the released beta maps. They are retained as separate stored outputs; their precise computational provenance should be checked before treating them as interchangeable.

### D. Sentence-onset reconstruction

```text
sentence_onset/reconstruction/sentence_onset_K20_reconstruction_summary.csv
sentence_onset/reconstruction/sentence_onset_K20_reconstruction_L.npy
sentence_onset/reconstruction/sentence_onset_K20_reconstruction_R.npy
```

These files document the reconstruction of sentence-onset cortical responses using the first 20 nonconstant cortical eigenmodes.

The reproduction notebook verifies stored reconstruction statistics against frozen numerical reference values.

### E. Long-wavelength concentration

```text
sentence_onset/concentration/corrected_concentration_inference_K20.csv
sentence_onset/concentration/corrected_concentration_inference_all_K.csv
```

These files quantify preferential allocation of sentence-onset responses to low-order cortical eigenmodes.

The analyses compare sentence-onset concentration with the empirical cortical modal-energy reference distribution.

### F. Geometry specificity

```text
geometry_specificity/cortical_vs_spherical_basis_R2.csv
geometry_specificity/onset_comparison/participant_onset_LB_vs_LOPO_PCA_bilateral.csv
geometry_specificity/folded_vs_sphere_functional_alignment/group_PCA_folded_vs_sphere.csv
geometry_specificity/spatial_null_eigenstrap/definitive_10000_bilateral_inference.csv
geometry_specificity/spatial_null_eigenstrap/definitive_10000_provenance.csv
```

These results support comparisons between folded cortical eigenmodes, spherical eigenmodes, and functional PCA representations.

LOPO denotes leave-one-participant-out functional PCA.

The spatial-null analysis evaluates whether functional-subspace alignment with folded cortical geometry exceeds alignment expected under spherical eigenmode surrogates.

This analysis is distinct from the temporal permutation tests used for sentence-onset validation.

## 3. Spatial eigenstrapping

The definitive spatial-null analysis uses:

- 20 population functional PCA components.
- A spherical FEM eigenmode basis containing 121 modes, including the constant mode.
- 11 spherical eigengroups.
- QR-based rotations within the surrogate construction.
- FEM-weighted subspace-overlap statistics.
- A primary comparison at K=20, with sensitivity analyses at K=5 and K=10.
- 10,000 surrogate realizations.
- A bilateral statistic averaging hemisphere-specific overlaps.

The curated release contains the definitive inference, provenance, and stored surrogate-derived results.

The canonical reproduction notebook verifies inference from these outputs. It does not regenerate the 10,000 spatial surrogates.

The original surrogate-generation implementation is not currently established as part of the tested lightweight reproduction route.

## 4. Additional geometry-specificity outputs

The repository also contains curated results for held-out variance capture using folded and spherical eigenmodes.

These include analyses under FEM-weighted and uniform spatial metrics.

The associated figure-generation script is:

```text
code/plot_figure_S3_geometry_specificity_FINAL.py
```

The held-out variance-capture analyses are separate from the sentence-onset reconstruction comparison and from the functional-PCA spatial-null test.

Their upstream analysis scripts require additional neuroimaging caches and geometric resources that are not included in the lightweight reproduction package.

## 5. Reproduced outputs

The canonical notebook writes its figures to:

```text
github_release/reproduced_figures/
```

The tested execution generates:

```text
empirical_cortical_energy_spectrum.png
sentence_onset_excess_concentration.png
spatial_null_overlap.png
```

The executed notebook is saved as:

```text
github_release/01_reproduce_paper_figures_executed.ipynb
```

In addition to generating these figures, the notebook validates numerical results for sentence-onset responses, reconstruction, concentration, and geometry specificity.

These outputs constitute a numerical reproduction and validation workflow, not a complete regeneration of all publication-ready manuscript figures.

## 6. Historical analyses

The historical `pang_out/` tree and several upstream notebooks contain analyses from earlier manuscript versions.

These include:

- sentence-shift, token-surprisal, and token-curvature analyses;
- four-effect eigenmode-profile comparisons and shape PCA;
- earlier cortical reconstruction figures;
- hippocampal and VTA analyses;
- exploratory linguistic predictors and alternative statistical models.

These materials may be useful for tracing the development of the project, but they are not canonical dependencies of the final manuscript reproduction notebook.

The presence of a historical output should not be interpreted as evidence that it contributes to the final reported results.

## 7. Reproducibility boundary

The public reproduction workflow begins with curated precomputed numerical results.

It does not, in one command, rerun:

- raw fMRI preprocessing;
- cortical eigenmode generation;
- language-model inference;
- subject- and run-level encoding models;
- generation of all temporal permutation distributions;
- generation of all spatial eigenstrapping surrogates;
- every supplementary figure-generation analysis.

The upstream scripts and notebooks provide additional methodological provenance where available.

The supported reproduction claim is therefore:

> The repository provides analysis code and a tested manuscript-level notebook that regenerates selected figures and verifies key numerical results from curated precomputed analysis outputs.
