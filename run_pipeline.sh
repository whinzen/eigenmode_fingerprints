#!/usr/bin/env bash
set -euo pipefail

BASE="${HOME}/eigenmode_fingerprints"
CODE="${BASE}/code"

echo "=== Eigenmode fingerprints pipeline ==="
echo "BASE: ${BASE}"

cd "${CODE}"

echo ""
echo "Step 1: Generate manuscript figures from precomputed outputs"

python plot_linguistic_composite_AB.py
python figure_sentence_level_empirical_reconstruction_twopanel.py
python figure_modes_plus_beta_pial.py
python make_empirical_reconstruction_summary_table.py
python analyze_reconstructed_map_correlations.py

echo ""
echo "Step 2: Optional supplementary reconstruction figures"

python figure_empirical_full_lowpass_residual.py --metric boundary --k-low 20
python figure_empirical_full_lowpass_residual.py --metric sentence_shift --k-low 20
python figure_empirical_full_lowpass_residual.py --metric token_shift --k-low 20
python figure_empirical_full_lowpass_residual.py --metric pred_error_ar --k-low 20
python figure_empirical_full_lowpass_residual.py --metric pred_error_subspace --k-low 20
python figure_empirical_full_lowpass_residual.py --metric curvature --k-low 20

echo ""
echo "Done. Figures written to:"
echo "${BASE}/pang_out/paper_figures"
echo "Tables written to:"
echo "${BASE}/pang_out/paper_tables"