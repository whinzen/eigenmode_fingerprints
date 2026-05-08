#!/bin/bash
# ================================================================
#  Setup script for a new independent "language_prg" project
#  Run this inside your pang_vtk environment on the cluster or locally.
# ================================================================

# --- Define root paths ---
SRC_DIR=~/eigenmode_fingerprints
DST_DIR=~/language_prg

echo "📁 Creating new project at $DST_DIR ..."

# --- Create directory hierarchy ---
mkdir -p $DST_DIR/{code,data/{embeddings,meta,external},results/{prg_runs,group_level,logs/prg},env,docs/figures}

# --- Copy or symlink embeddings from eigenmode project ---
# (choose ONE of the next two lines)

# Option A: copy the files (safe, editable)
cp -v $SRC_DIR/pang_out/embeddings/en_run*_layer9.npy $DST_DIR/data/embeddings/

# Option B: create symlinks (saves disk space)
# ln -s $SRC_DIR/pang_out/embeddings/en_run*_layer9.npy $DST_DIR/data/embeddings/

# --- Copy your existing PRG analysis code ---
cp -v $SRC_DIR/code/language_prg_temporal.py $DST_DIR/code/

# --- Create placeholder files ---
cat > $DST_DIR/docs/README.md <<'EOF'
# Language PRG Project

This project analyzes scale-dependent correlations in contextual language embeddings
using a phenomenological renormalisation group (PRG) approach.

Run individual analyses via `language_prg_temporal.py`, and aggregate results using
`group_prg_aggregate.py` once per-run outputs are generated.
EOF

cat > $DST_DIR/env/environment.yml <<'EOF'
name: pang_vtk
channels:
  - conda-forge
dependencies:
  - python>=3.10
  - numpy
  - scipy
  - matplotlib
  - scikit-learn
  - pandas
EOF

# --- Add basic run script templates ---
cat > $DST_DIR/code/run_all_prg.sh <<'EOF'
#!/bin/bash
# Launch PRG analyses for all English runs (15–23)

for RUN in {15..23}; do
  sbatch ~/language_prg/code/sbatch_prg_run.sbatch $RUN
done
EOF
chmod +x $DST_DIR/code/run_all_prg.sh

cat > $DST_DIR/code/sbatch_prg_run.sbatch <<'EOF'
#!/bin/bash
#SBATCH -J prg_en
#SBATCH -A your_account
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH --mem=16G
#SBATCH -o ~/language_prg/results/logs/prg/%x_%j.out
#SBATCH -e ~/language_prg/results/logs/prg/%x_%j.err

source activate pang_vtk

RUN=$1
EMB=~/language_prg/data/embeddings/en_run${RUN}_layer9.npy
OUT=~/language_prg/results/prg_runs/en_run${RUN}_layer9
mkdir -p $OUT

python ~/language_prg/code/language_prg_temporal.py \
    --embeddings $EMB \
    --scales 1,2,4,8,16,32 \
    --overlap 8 \
    --max-lag 128 \
    --outdir $OUT \
    --prefix prg_lang \
    --save-mats
EOF

echo "✅ Project folders and templates created in $DST_DIR"
echo "To check structure:"
echo "  tree -L 3 $DST_DIR"
echo
echo "Then activate your environment:"
echo "  conda activate pang_vtk"
echo "and launch one test run:"
echo "  sbatch code/sbatch_prg_run.sbatch 15"