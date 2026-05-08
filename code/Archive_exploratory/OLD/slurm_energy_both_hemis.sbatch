#!/bin/bash
#SBATCH --job-name=energy_both
#SBATCH --output=logs/energy_both_%a.out
#SBATCH --error=logs/energy_both_%a.err
#SBATCH --array=15-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition=short

source ~/project/anaconda3/etc/profile.d/conda.sh
conda activate pang_vtk

cd /gpfs/home/whinzen/eigenmode_fingerprints/code

# Run both-hemisphere energy extraction
python energy_per_run_timeseries.py --run_id $SLURM_ARRAY_TASK_ID