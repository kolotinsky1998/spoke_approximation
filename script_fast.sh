#!/bin/bash
#SBATCH --job-name=spoke_approx
#SBATCH --error=err_spoke_approx
#SBATCH --output=output_spoke_approx
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1                
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH -p cpu-e-quick
#SBATCH -A proj_1776

srun python -u colab_scripts/fit_spoke_pysr.py --average-file outputs/rotating_average/rotating_average.npz --metadata outputs/rotating_average/metadata.json --out-dir outputs/pysr_polar
