#!/bin/bash

#SBATCH -e output/simulation_%A_%a.err
#SBATCH -o output/simulation_%A_%a.out
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH --array=1-500

# Load the require modules
module load Gurobi/8.11


# source activate bandit_surrogates 

python ./scripts/gen_simulations.py

