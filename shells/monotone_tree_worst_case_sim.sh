#!/bin/bash
#SBATCH -e output/worst_case.err
#SBATCH -o output/worst_case.out
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem=128GB

# Load the require modules
module load Gurobi/8.11


source activate bandit_surrogates 

python ./scripts/monotonic_tree_worst_case_sim.py

