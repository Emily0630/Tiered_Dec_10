#!/bin/bash
#SBATCH -e output/reg_comp_%A_%a.err
#SBATCH -o output/reg_comp_%A_%a.out
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem-per-cpu=32G
#SBATCH --array=1-6

# Load the require modules
module load Gurobi/8.11


source activate bandit_surrogates 

python ./scripts/regression_simulations.py

