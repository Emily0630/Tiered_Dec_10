# library(tidyverse)
# library(mvtnorm)
# library(foreach)
# library(parallel)

source("./R/r_unit_dball.R")
source("./R/dominance.R")
source("./R/orders.R")
source("./R/estimators.R")

args <- commandArgs(TRUE)


surrogate_path <- args[1]
measure_path <- args[2]

if (!file.exists(surrogate_path)) {
  cat('Cannot find', surrogate_path, 'exiting!\n')
  stop()
}

if (!file.exists(measure_path)) {
  cat('Cannot find', measure_path, 'exiting!\n')
  stop()
}



Z <- data.matrix(read.csv(surrogate_path))
Qs <- data.matrix(read.csv(measure_path))

n <- as.integer(args[3])
mod <- as.integer(args[4])

cat("mod = ", mod, "\n")
cat("n = ", n, "\n")
cat(dim(Z), "\n")
cat(dim(Qs), "\n")
cat("Generating Relation List\n")
RL <- relation_list(Z, p_order = product_order)
cat("Generating Sparse Constraints\n")
constrs <- constraints_sprs(RL, n)

# grab the array id value from the environment variable passed from sbatch
taskID <- as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
cat(taskID, "\n")

indices <- c(taskID:(taskID+ mod-1))

#cat(paste0(indices, sep = ", "), "\n")
dom_id <- sapply(indices, function (i) dominated(id = i, Qs = Qs, constr = constrs, solver = "glpk"))

non_dominated <- indices[which(dom_id==0)]

cat("Writing data\n")
folder <-paste0(getwd(), "/results/nodes")
if (!file.exists(folder)) {
  dir.create(folder)
}
outfile = paste0(getwd(), "/results/nodes/non_dominated_", sprintf("%02d", taskID),".csv")

## Write using array_id
write.table(non_dominated, outfile, sep=',', row.names=FALSE, quote=FALSE, col.names=FALSE)

