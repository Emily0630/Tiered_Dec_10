#!/bin/bash
#SBATCH -e output/reg_comp.err
#SBATCH -o output/reg_comp.out
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem-per-cpu=32G

# Load the require modules
module load Gurobi/8.11


# source activate bandit_surrogates 

python ./scripts/regression_simulations.py

