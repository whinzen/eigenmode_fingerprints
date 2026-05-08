#!/bin/bash
#SBATCH --job-name=testEN057
#SBATCH -p high
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/eigenmode_fingerprints/logs/testEN057_%j.out
#SBATCH --error=$HOME/eigenmode_fingerprints/logs/testEN057_%j.err

set -euo pipefail

PROJ="$HOME/eigenmode_fingerprints"
DERIV="$PROJ/ds003643/derivatives"
STIM="$PROJ/ds003643/annotation/EN"
WARPED="$PROJ/warped_fsavg"
OUT="$PROJ/outputs_en"
FSAVG="$PROJ/fsavg_brain.nii.gz"

SUB=sub-EN057

echo "=== $(date) :: node $(hostname) :: $SUB ==="

module load Miniconda3 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
conda activate eigenmodes

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4

mkdir -p "$WARPED" "$OUT" "$PROJ/logs"

python "$PROJ/code/warp_to_fsaverage.py" \
  --deriv "$DERIV" \
  --out   "$WARPED" \
  --fsavg_brain "$FSAVG" \
  --transform Affine \
  --pattern "$SUB/func/*task-lppEN*desc-preproc_bold.nii.gz"

python "$PROJ/code/rename_warped_fsavg.py" \
  --src-deriv "$DERIV" \
  --warped    "$WARPED" \
  --subject   "$SUB"

python "$PROJ/code/fix_tr_in_warped.py" \
  --src-deriv "$DERIV" \
  --warped    "$WARPED" \
  --subject   "$SUB"

python -m code.lpp_batch_runner \
  --deriv "$WARPED/$SUB" \
  --stim  "$STIM" \
  --out   "$OUT" \
  --fsaverage fsaverage5 \
  --kmodes 120

echo "=== $(date) :: done $SUB ==="