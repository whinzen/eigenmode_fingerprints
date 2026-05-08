#!/bin/bash
set -euo pipefail

# Project paths
PROJ="/Users/whinzen/Dropbox/My/My projects/eigenmode_fingerprints"
DERIV="$PROJ/ds003643/derivatives"
STIM="$PROJ/ds003643/annotation/EN"
WARPED="$PROJ/warped_fsavg"
OUT="$PROJ/outputs_en"
FSAVG="$HOME/freesurfer/fsavg_brain.nii.gz"

# List of English subjects (from derivatives folder)
SUBJECTS=$(cd "$DERIV" && ls -d sub-EN* | sort)

for SUB in $SUBJECTS; do
  echo "=== Processing $SUB ==="

  # 1) Warp this subject’s runs
  python "$PROJ/code/warp_to_fsaverage.py" \
    --deriv "$DERIV" \
    --out "$WARPED" \
    --fsavg_brain "$FSAVG" \
    --transform Affine \
    --pattern "$SUB/func/*task-lppEN*desc-preproc_bold.nii.gz"

  # 2) Rename warped runs to BIDS-style
  python "$PROJ/code/rename_warped_fsavg.py" \
    --src-deriv "$DERIV" \
    --warped "$WARPED" \
    --subject "$SUB"

  # 3) Run eigenmode pipeline
  python -m code.lpp_batch_runner \
    --deriv "$WARPED/$SUB" \
    --stim "$STIM" \
    --out "$OUT" \
    --fsaverage fsaverage5 \
    --kmodes 120
done