#!/bin/bash
#SBATCH --job-name=lppEN_one
#SBATCH -p high
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- config you can tweak ---
SUB=${SUB:-sub-EN057}
DERIV="$HOME/eigenmode_fingerprints/ds003643_derivs"
PROJ="$HOME/eigenmode_fingerprints"
FSAVG="$PROJ/fsavg_brain.nii"         # or .nii.gz, adjust if needed
OUTROOT="$PROJ"
FSLEVEL="fsaverage5"
KMODES=120
# ----------------------------

echo "=== $(date) :: node $(hostname) :: ${SUB} ==="
echo "Python: $(which python)"; python --version || true

mkdir -p "$PROJ/code" "$PROJ/modes/$FSLEVEL" "$PROJ/warped_fsavg" "$PROJ/outputs_en"

echo "[1/4] Warp to fsaverage (ANTS)"
python "$PROJ/code/warp_to_fsaverage.py" \
  --deriv "$DERIV" \
  --out   "$PROJ/warped_fsavg" \
  --fsavg_brain "$FSAVG" \
  --transform Affine \
  --pattern "$SUB/func/*task-lppEN*space-MNIColin27*desc-preproc_bold.nii.gz"

echo "[2/4] Fix TR metadata"
python "$PROJ/code/fix_tr_in_warped.py" \
  --src-deriv "$DERIV" \
  --warped    "$PROJ/warped_fsavg" \
  --subject   "$SUB"

echo "[3/4] Make/Cache fsaverage5 eigenmodes (if not exist)"
python "$PROJ/code/compute_fsaverage5_modes.py" \
  --outdir "$PROJ/modes/$FSLEVEL" \
  --k $KMODES

echo "[4/4] Project each run to surface + eigenmodes"
python "$PROJ/code/project_and_energy_fs5.py" \
  --warped   "$PROJ/warped_fsavg/$SUB" \
  --modes    "$PROJ/modes/$FSLEVEL" \
  --out      "$PROJ/outputs_en" \
  --kmodes   $KMODES

echo "=== DONE $(date) ==="