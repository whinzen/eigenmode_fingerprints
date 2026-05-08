#!/bin/bash
#SBATCH --job-name=energy_both_hemi
#SBATCH --output=logs/energy_both_%a.out
#SBATCH --error=logs/energy_both_%a.err
#SBATCH --array=15-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Load environment
source ~/project/anaconda3/bin/activate pang_vtk

# Move to code directory
cd /gpfs/home/whinzen/eigenmode_fingerprints/code

# Run energy computation script with hemisphere overwrite for this run
python energy_per_run_timeseries.py --run_id $SLURM_ARRAY_TASK_ID