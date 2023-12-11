##
## @file test_sim_slurm.R
## @brief Testing simulation on DCC
## @author Marc Brooks
##

#library(tidyverse)
library(mvtnorm)

source("./R/r_unit_dball.R")
# source("./R/dominance.R")
# source("./R/orders.R")
source("./R/estimators.R")
source("./R/generative_model.R")

# Set up
if (!file.exists('./results')) {
  dir.create("./results")
}

if (!file.exists('./results/nodes')) {
  dir.create("./results/nodes")
}

if (!file.exists('./data')) {
  dir.create("./data")
}

if (!file.exists('./output')) {
  dir.create("./output")
}



# Data parameters ----
n <- 100 # for simulation study
p <- 15 # dimension of covariates + 1 for the intercept
q <- 3  # for simulation study
rho <- .5

# Generating X values ----
px <- p-1
# px <- p
Sigma_X <- matrix(numeric(px^2), ncol=px)
Sigma_X <- outer(1:nrow(Sigma_X), 1:ncol(Sigma_X) ,
                 FUN=function(r,c) rho^(abs(r-c)))
X_mu <- rep(0, px)

sample_X <- \() cbind(rep(1,n) ,rmvnorm(n = n, X_mu, Sigma_X))

#sample_X <- \() rmvnorm(n = n, X_mu, Sigma_X)

# Setting actions ----
K = 2
A = 1:K - 1


# Get thetas ----
sample_theta <- function(i) rbind(rep(10*(i), q) ,  matrix(rnorm(q*px), nrow=px))
#sample_theta <- function(i) matrix(rnorm(q*p, mean = 0, sd = (i+1*(1-i) + 4*(i)) ), nrow=px)
theta <- lapply(A, sample_theta)


# Primary outcome ----
# monotone spline ----
L <- 10

g <- gen_g_monotone(q,L)

get_Y <- \(z) get_outcome(Z=z, g=g, delta_mu = 1, delta_sig2 = 5, eta_mu = 0, eta_sig2 = 10)


# Begin Simulation ----
# Create policy class ----
N = 10000
gammas <- r_unit_dball(d=p, n=N)

PI <- lapply(seq_len(nrow(gammas)), function(i) \(x) lin_pi(x, gammas[i, ]))

# Decision points ----
# Playing around with single iteration

Tot <- 1
# Populate history
varnames_X <- paste0("X", 1:p)
varnames_Z <- paste0("Z", 1:q)
varnames <- c("pid", varnames_X, varnames_Z, "Y", "a", "propensity", "t")

# H <- matrix(0, nrow = Tot*n, ncol <- 1 + 1 + p + q + 1 + 1 + 1 + 1)
H <- matrix(0, nrow = Tot*n, ncol <- 1 +  p + q + 1 + 1 + 1 + 1)
colnames(H) <- varnames
H[,"t"] <- Inf
H[ , "pid"] <- rep(1:n, Tot)



H[,'a'] <- sample(A, size=n, replace=T)
H[,"propensity"] <- 1/length(A)
H[ ,varnames_X] <- sample_X()
for (i in seq_len(n)) {
  a_i <- H[i, 'a']
  H[i, varnames_Z] <- sample_Z(X = H[i ,varnames_X], theta = theta[[a_i + 1]])
}
for (i in seq_len(n)) {
  H[i, "Y"] <- get_Y(H[i, varnames_Z])
}

Qs <- do.call(cbind, lapply(seq_along(PI), function (i) {
  q_measure(pi =  PI[[i]],
            X = H[,varnames_X],
            a = H[ , 'a'],
            propensity = H[, "propensity"])
}))


surrogate_path <- paste0(getwd(), '/data/', 'surrogates.csv')
measure_path <- paste0(getwd(), '/data/', 'measure.csv')
# data_path <- paste0(getwd(), '/data/', 'data.csv')

write.csv(H[, varnames_Z], surrogate_path, row.names = F)
write.csv(Qs, measure_path, row.names = F)

# write.csv(H, data_path, row.names = F)

jobs <- N # number of policies
mod <- 20 #




sink("dominance.sh")
cat("#!/bin/bash\n")
cat("#SBATCH -e output/test_%A_%a.err\n")
cat("#SBATCH -o output/test_%A_%a.out\n")
cat("#SBATCH --mem=2000\n")
cat("#SBATCH --array=1-",jobs, ":", mod, "\n\n", sep = "")
cat("module load GLPK/5.0\n")
cat("module load R/4.1.1-rhel8\n")
cat("Rscript ./R/non_dominated_regimes.R", surrogate_path, measure_path, n,   mod, sep = " ")
sink()

# Submit to run on cluster
cat("Starting slurm job array for dominated regimes!\n")
system(paste("sbatch", "dominance.sh"))


statrtime <- Sys.time()
nfiles <- jobs/mod
while(T) {
  if (length(list.files('./results/nodes')) == nfiles) {
    filenames <- list.files('./results/nodes', full.names = T)
    non_dom_regimes <- unlist(lapply(filenames, function(name) read.csv(name)[[1]]))
    break
  }
}
cat(Sys.time() - statrtime)


outfile <- paste0(getwd(),'/results/', 'non_dominated_regimes.csv')
cat("Writing non dominated regimes\n")
write.table(non_dom_regimes, outfile, sep=',', row.names=FALSE, quote=FALSE, col.names=FALSE)
