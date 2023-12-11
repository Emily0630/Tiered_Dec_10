#!/bin/bash
#SBATCH -e output/test_%A_%a.err
#SBATCH -o output/test_%A_%a.out
#SBATCH --mem=2000
#SBATCH --array=1-10000:10

module load GLPK/5.0
module load R/4.1.1-rhel8
Rscript ./R/non_dominated_regimes.R /hpc/group/laberlabs/mgb45/data/surrogates.csv /hpc/group/laberlabs/mgb45/data/measure.csv 100 10