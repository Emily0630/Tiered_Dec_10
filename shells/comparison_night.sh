#!/bin/bash
#SBATCH -p scavenger
#SBATCH -e output/monotone_tree_12.err
#SBATCH -o output/monotone_tree_12.out
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=128G

# Load the require modules
# module load Gurobi/8.11

# export PYTHONPATH="${PYTHONPATH}:/hpc/home/yl880/cb_surrogates"
export PYTHONPATH="${PYTHONPATH}:/Users/yinyihongliu/Desktop/pythonProject1"

# source activate bandit_surrogates 

python ./scripts/comparisons_test_bandit_night.py

