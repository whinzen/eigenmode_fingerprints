#!/bin/bash
#SBATCH --job-name=energy_all
#SBATCH --output=logs/energy_all.out
#SBATCH --error=logs/energy_all.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --partition=short

source activate pang_vtk

cd /gpfs/home/whinzen/eigenmode_fingerprints/code

python energy_compute_per_run.py
