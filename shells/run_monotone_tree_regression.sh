#!/bin/bash
#SBATCH -e output/monotone_tree_%A.err
#SBATCH -o output/monotone_tree_%A.out
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem=64GB

# Load the require modules
module load Gurobi/8.11


source activate bandit_surrogates 

export p=20
export q=5
export n_trees=100
export n_actions=36
export n_samples=100

if [ ! -f "joblist" ]; then
       touch "joblist" 2>/dev/null
   fi
   echo $SLURM_JOB_ID, $p, $q, $n_trees, $n_actions, $n_samples >> joblist

python ./scripts/run_monotone_tree_regression.py --n_samples $n_samples --n_trees $n_trees --n_actions $n_actions --p $p --q $q

