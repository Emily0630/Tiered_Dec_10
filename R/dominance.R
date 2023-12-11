##
## @file dominance.R
## @brief Evalaute whether one policy is dominated by another
## @author Marc Brooks
## @note Written hastily and rather poorly
##

library(lpSolveAPI)
library(Rglpk)

#library(parallel)
#library(doParallel)

# Local packages
source("./R/orders.R")

# registerDoParallel(cores = detectCores())

setup_lp <- function(R) {

  diag(R) <- 0
  n <- ncol(R)
  for (i in seq_len(n)) {
    for (j in i:n) {
      if ((R[i,j] == 1) & (R[j,i] == 1)) {
        R[i,j] <- R[j, i] <- 0
      }
    }
  }

  z <- sum(R)
  m <- n + z + 1 # number of constraints, n (each ai <=1, m (number sum(R) comparable outcomes), 1 for sum a_i = 1)
  lprec <- make.lp(m,n)

  k <- 1

  for (i in seq_len(n)) {
    idx <- which(R[i,] ==1)
    for (j in idx) {
      set.row(lprec, k, c(1,-1), indices = c(i,j))
      # set.column(lprec, j, -1, indices = k)

      k = k + 1
    }
    set.row(lprec, row = z + i, 1, i)

  }
  set.row(lprec, row=m, rep(1,n), indices = c(1:n))

  set.constr.type(lprec, c(rep("<=", m-1), "="))
  set.rhs(lprec, c(rep(0, z), rep(1, n+1)))


  return(lprec)
}


lp_from_relation <- function(RL, n) {
  # Constructs lp problem for lpApiSolve solver
  # RL is a relation list
  # n is for the number of variables
  RL <- cover_relation_list(RL)

  z <- nrow(RL)
  m <- n + z + 1 # number of constraints, n (each ai <=1, m (number sum(R) comparable outcomes), 1 for sum a_i = 1)

  # make lp
  lprec <- make.lp(m,n)
  for (i in seq_len(z)) {
    set.row(lprec, i, c(1,-1), indices = RL[i, ])
  }

  for (i in seq_len(n)) {
    set.row(lprec, row = z + i, 1, i)

  }

  set.row(lprec, row=m, rep(1,n), indices = c(1:n))

  set.constr.type(lprec, c(rep("<=", m-1), "="))
  set.rhs(lprec, c(rep(0, z), rep(1, n+1)))

  return(lprec)
}

constraints_sprs <- function(RL, n) {
  # generates constraints for glpk solver
  RL <- cover_relation_list(RL)
  z <- nrow(RL)
  m <- n + z + 1 # number of constraints, n (each ai <=1, m (number sum(R) comparable outcomes), 1 for sum a_i = 1)
  mat <- simple_triplet_zero_matrix(nrow = m, ncol = n)

  for (i in seq_len(z)) {
    # constraints if z_i <= z_j we want a_i - a_j <= 0
    mat[i, RL[i, 1]] <- 1
    mat[i, RL[i, 2]] <- -1
  }

  for (i in seq_len(n)) {
    mat[z + i, i] <- 1 # constrain that 0 <= ai <= 1
    mat[m, i] <- 1 # final constaint Sum a_i = 1
  }

  return(list("constr_mat" = mat,
              "constr.dir" = c(rep("<=", m-1), "=="),
              "rhs" = c(rep(0, z), rep(1, n+1))
                ))

}

constraints <- function(R){
  # Take relation matrix from partial order to construct constrain matrix and right hand side
  # for LP formulation

 diag(R) <- 0
 m <- ncol(R)
 for (i in seq_len(m)) {
   for (j in i:m) {
     if ((R[i,j] == 1) & (R[j,i] == 1)) {
       R[i,j] <- R[j, i] <- 0
     }
   }
 }
 n <- sum(R)
 constr_mat <- matrix(0, n + m, m)
 k = 1
 for (i in seq_len(m)) {
   idx <- which(R[i,] ==1)
   for (j in idx) {
     constr_mat[k, i] = 1
     constr_mat[k, j] = -1
     k = k + 1
   }
   constr_mat[n+i, i] = 1
 }

 return(list("constr_mat" = constr_mat,
             "constr.dir" = rep("<=", n+m),
             "rhs" = c(rep(0, n), rep(1, m))))
}


dominance_sprs <- function(q1, q2, lp_model) {
  # Solves LP to determine which of p or q is dominated
  obj_fun <- q2 - q1
  lp.control(lp_model, sense = "min")
  set.objfn(lp_model, obj = obj_fun)
  solve(lp_model)
  val_min <- get.objective(lp_model)

  lp.control(lp_model, sense = "max")
  solve(lp_model)
  val_max <- get.objective(lp_model)
  # Return 1 for q1 <- and 0 for not.
  return(1*(val_min >= 0 & val_max > 0))

}


dominance_old <- function(q1, q2, constrs) {
  # Solves LP to determine which of p or q is dominated
  stopifnot(all(names(constrs) == c("constr_mat", "constr.dir", "rhs")))
  obj.fun <- q2 - q1

  min_sol <- lp("min",
            objective.in = obj.fun,
            const.mat = constrs$constr_mat,
            const.dir = constrs$constr.dir,
            const.rhs = constrs$rhs
            )

  max_sol <- lp("max",
                objective.in = obj.fun,
                const.mat = constrs$constr_mat,
                const.dir = constrs$constr.dir,
                const.rhs = constrs$rhs
  )

  val_min <- min_sol$objval
  val_max <- max_sol$objval

  # Return 1 for q1 <- and 0 for not.
  return(1*(val_min >= 0 & val_max > 0))
}



dominance_glpk <- function(q1,q2, constr) {
  # Soliving if policiy is dominated using glpk solver

  obj = q2 - q1

  min_lp <- Rglpk_solve_LP(obj,
                           mat = constr$constr_mat,
                           dir = constr$constr.dir,
                           rhs = constr$rhs
  )

  max_lp <- Rglpk_solve_LP(obj,
                           mat = constr$constr_mat,
                           dir = constr$constr.dir,
                           rhs = constr$rhs,
                           max=T
  )
  return(1*(min_lp$optimum >=0) & (max_lp$optimum >0))

}


dominated <- function(id, Qs, lp_model=NULL, constr=NULL, solver = "glpk") {
  # Compares surrogate given by policy to other measures indcued by other policies in class
  stopifnot(solver %in% c("glpk", "lpsolve"))

  if (solver=="glpk") {
    stopifnot(class(constr) == "list")
    stopifnot(class(constr$constr_mat) == "simple_triplet_matrix")

    dominance <- \(q2) dominance_glpk(q1=Qs[ ,id], q2=q2, constr = constr)
  } else if (solver == "lpsolve") {
    stopifnot(class(lp_model) == "lpExtPtr")
    dominance <- \(q2) dominance_sprs(q1=Qs[ ,id], q2=q2, lp_model = lp_model)
  }

  cand_id <- c(1:ncol(Qs))[-id]
  res <- 0
  for (i in cand_id) {
    dom <- dominance(Qs[,i])
    if (dom == 1) {
      return(1)
    }
  }
  return(res)
}

non_dominated_class <- function(Qs, lp_model=NULL, constr=NULL, solver = "glpk", ncores=1) {

  stopifnot(solver %in% c("glpk", "lpsolve"))

  if (solver=="glpk") {
    stopifnot(class(constr) == "list")
    stopifnot(class(constr$constr_mat) == "simple_triplet_matrix")

    dom_fcn <- \(i) dominated(i, Qs=Qs, constr = constr, solver = "glpk")

  } else if (solver == "lpsolve") {
    stopifnot(class(lp_model) == "lpExtPtr")

    dom_fcn <- \(i) dominated(i, Qs=Qs, lp_model = lp_model, solver = "lpsolve")
  }

  n <- ncol(Qs)
  res <- do.call(rbind,
                 mclapply(1:n, function(i) c(i, dom_fcn(i)), mc.cores = ncores)
                 )
  return(res[which(res[,2]==0), 1])
}


# # Test
# Z <- matrix(rnorm(20), ncol=2)
# R <- relation_matrix(Z, partial_order2)
#
# constrs <- constraints(R)
# q1 = runif(10)
# q2 = runif(10)
#
# q1_q2 = q1 - q2
#
# dom <- dominance(q1,q2, constrs)


## Test 1
# z1 <- c(1 ,2)
# z2 <- c(3 ,2)
# z3 <- c(1 ,5)
# z4 <- c(5 ,5)
# z5 <- c(3 ,3)
# Z <- rbind(z1,z2,z3,z4,z5)
# R <- relation_matrix(Z, product_order)
# rl <- relation_list(Z, product_order)
# covR <- cover_matrix(rl)
#
# q1 <- c(0, .1, 0, .6,.3)
# q2 <- rep(1,5)/5
#
# constr <- constraints(R)
# constr2 <- constraints(covR)
#
#
# lpfromRL <- lp_from_relation(RL = rl, n = ncol(R))
#
# # lp1 <- setup_lp(R)
# set.objfn(lpfromRL, q1-q2)
# solve(lpfromRL)
# get.objective(lpfromRL)
# lp.control(lpfromRL, sense="max")
# solve(lpfromRL)
# get.objective(lpfromRL)
#
#
# dominance_sprs(q1,q2, lp_model = lpfromRL)
# dominance_sprs(q2,q1, lp_model = lpfromRL)

#
# dom1 <- dominance(q1, q2, constrs = constr)
# dom2 <- dominance(q2, q1, constrs = constr)
#
#
# sol1 <- lp("min",
#           objective.in = q1 - q2,
#           const.mat = constr$constr_mat,
#           const.dir = constr$constr.dir,
#           const.rhs = constr$rhs,
#           compute.sens = T
# )
#
# sol2 <- lp("min",
#            objective.in = q2 - q1,
#            const.mat = constr$constr_mat,
#            const.dir = constr$constr.dir,
#            const.rhs = constr$rhs,
#            compute.sens = T
# )
#
#
## Test 2
#
# Z <- matrix(rnorm(30), ncol=2)
# R <- relation_matrix(Z, product_order)
#
# myLP <- setup_lp(R)
# constr <- constraints(R)
#
# p = .5
# Qs <- do.call(
#   cbind,
#       lapply(1:100, function(i) {
#       if (p < runif(1)) {
#         runif(15,.5,1)
#       } else {
#         runif(15,0,.6)
#       }
#         }
#       )
# )

# # Checking that if a distribution is doesn't dominate a distribution that dominates it.
# doms1 <- sapply(2:100, function(i) dominance(Qs[,1], Qs[,i], constrs = constr))
# idx <- which(doms1==1)
# sapply(idx, function(i) dominance(Qs[,i], Qs[,1], constrs = constr))
#
# non_doms2 <- non_dominated_class(Qs = Qs,constrs = constr)
# non_doms <- non_dominated_class(Qs = Qs,constrs = constr)
# sapply(non_doms, function(i) dominated(i, Qs, constr))
