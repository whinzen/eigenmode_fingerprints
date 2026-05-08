#!/bin/bash
#SBATCH --job-name=qwen_embed_gpu
#SBATCH --output=/gpfs/home/whinzen/eigenmode_fingerprints/code/logs/qwen_embed_gpu_%j.out
#SBATCH --error=/gpfs/home/whinzen/eigenmode_fingerprints/code/logs/qwen_embed_gpu_%j.err
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

mkdir -p /gpfs/home/whinzen/eigenmode_fingerprints/code/logs
mkdir -p /gpfs/home/whinzen/eigenmode_fingerprints/pang_out/token_embeddings_qwen

source /home/whinzen/project/anaconda3/etc/profile.d/conda.sh
conda activate pang_vtk

cd /gpfs/home/whinzen/eigenmode_fingerprints/code

echo "=== Job info ==="
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-none}"
which python
python --version

echo "=== CUDA check ==="
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
EOF

echo "=== Starting Qwen embedding extraction ==="
python -u build_token_embeddings_qwen.py \
  --model Qwen/Qwen3-0.6B \
  --layer -4 \
  --batch-size 16 \
  --max-length 256 \
  --out-name token_embeddings_qwen3_0p6b.pkl

echo "=== Done ==="
date
ls -lh /gpfs/home/whinzen/eigenmode_fingerprints/pang_out/token_embeddings_qwen/