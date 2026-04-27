# Eigenmode fingerprints of linguistic structure in cortex

This repository contains code to reproduce the eigenmode analyses, GLMs, control analyses, and paper figures for the manuscript.

## Overview

The analysis projects fsaverage5 cortical BOLD data onto Laplace–Beltrami eigenmodes of the cortical surface. Mode-wise energy spectra are used to test scale-free organization. Linguistic and geometric regressors are then used in GLMs to estimate how sentence-level and token-level language variables modulate eigenmode energy.

---

## Required inputs

Expected project layout:

```text
eigenmode_fingerprints/
├── code/
├── modes/fsaverage5/
│   ├── L_phi.npy
│   ├── R_phi.npy
│   ├── lam_L.npy
│   └── lam_R.npy
├── data/empirical/sub-*/func/
│   └── *_hemi-*_space-fsaverage5_bold.func.gii
├── ds003643/annotation/EN/repunct/
│   ├── lppEN.csv
│   └── little_prince_series_curvature.csv
└── pang_out/
```

The repository assumes preprocessed surface BOLD files in fsaverage5 space and precomputed cortical eigenmodes.

---

## Environment

Main Python dependencies:

- numpy  
- pandas  
- scipy  
- matplotlib  
- nibabel  
- nilearn  
- neuromaps  

### Connectome Workbench (required for Gradient 1 comparison)

```bash
export PATH=~/tools/workbench/bin_linux64:$PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Check installation:

```bash
wb_command -version
```

---

## Full pipeline

Run:

```bash
bash run_pipeline.sh
```

This executes the analysis in the following order.

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

Outputs include per-run mode coefficients, energy spectra, group spectra, criticality fits, and diagnostic tables.

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

These generate subject/run-aligned sentence-boundary and sentence-shift regressors, run eigenmode GLMs, and aggregate effects across subjects.

---

## 3. Token-level transition metrics

Scripts:

```text
analyze_word_transition_geometry.py
build_transition_metric_regressors.py
rebuild_glm_transition_metric_per_run.py
group_aggregate_transition_metric.py
```

These compute token-level transition metrics from contextual embeddings and estimate their eigenmode profiles.

---

## 4. Curvature-family analyses

Scripts:

```text
build_curvature_regressors_per_subject.py
rebuild_glm_curvature_per_run.py
group_aggregate_curvature.py
figure_curvature_family_collapse.py
```

These use externally provided curvature-family metrics from `little_prince_series_curvature.csv`, including global curvature, path length, chord length, and mean turning angle.

---

## 5. Word-rate controls

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

These generate TR-aligned word-rate regressors and estimate both univariate and joint/residualized control models.

---

## 6. Paper figures and tables

Scripts:

```text
figure_modes_plus_beta_pial.py
plot_cumulative_bold_reconstruction_variance.py
compare_mode3_to_gradient1_template.py
make_table1_sentence_level.py
make_table2_token_level.py
make_table3_transition_metrics_eigenmodes.py
```

Key outputs are written to:

```text
pang_out/paper_figures/
pang_out/paper_tables/
```

---

## Notes

### Eigenmode indexing

Mode 0 is the near-constant mode and is excluded from most interpretive analyses. Reported effects focus on non-constant low-order modes, especially mode 3.

### Reconstructed cortical maps

Reconstructed maps are not raw activation maps. They are low-dimensional projections obtained by combining eigenmodes with their GLM beta weights:

$begin:math:display$
\\hat\{m\}\(v\) \= \\sum\_\{k\=1\}\^\{K\} \\beta\_k \\phi\_k\(v\)
$end:math:display$

where $begin:math:text$\\phi\_k\(v\)$end:math:text$ is eigenmode $begin:math:text$k$end:math:text$ at cortical vertex $begin:math:text$v$end:math:text$.

### Cumulative energy

The cumulative energy figure reports the fraction of retained eigenmode coefficient energy captured by the first $begin:math:text$K$end:math:text$ modes. Because only the retained eigenmode subspace is used, this reflects variance within that subspace rather than full vertexwise BOLD variance.

---

## Reproducibility workflow

- Full pipeline:

```bash
bash run_pipeline.sh
```

- Figures only:

```text
notebooks/01_reproduce_paper_figures.ipynb
```
