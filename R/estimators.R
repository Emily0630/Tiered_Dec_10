##
## @file estimators.R
## @brief DTR estimators
## @author Marc Brooks
## @note Written hastily and rather poorly
##

library(rlist)

# Linear policy class
lin_pi <- function(X, gamma) {
  1*(sign(X %*% gamma) > 0)
}


# q_measure_old <- function(pi, X, t_index, a, propensity) {
#   # Takes
#   #  pi: policy function that takes data and returns action
#   #  X: covariate history
#   #  t_index: time index (decision points) that correspond to data and actions
#   #  a: history of actions
#   # propensity: history of probability action was assigned
#
#   t <- max(t_index)
#   numerator <- (a[t_index == t] == pi(X[t_index == t, ]))/propensity[t_index == t]
#
#   denominator <- sum((a == pi(X))/propensity)
#   return(numerator/denominator)
# }

q_measure <- function(pi, X, a, propensity) {
  # Takes
  #  pi: policy function that takes data and returns action
  #  X: covariate history
  #  t_index: time index (decision points) that correspond to data and actions
  #  a: history of actions
  # propensity: history of probability action was assigned

  numerator <- (a == pi(X))/propensity

  return(numerator/sum(numerator))
}

ipw <- function(pi, X, Y, a, propensity) {
  # IPW estimator for Value of policy pi with respect to Y
  # Takes
  #  pi: policy function that takes data and returns action
  #  X: covariate history
  #  a: history of actions
  #  propensity: history of probability action was assigned

  numerator <- Y*(a == pi(X))/propensity
  denominator <- sum((a == pi(X))/propensity)

  return(sum(numerator/denominator))
}



eps_greedy <- function(bandit, t) {

  # PI is a list of policies ( functions of  X)
  PI <- bandit$PI
  varnames_X <- bandit$varnames_X
  n <- bandit$n
  K <- bandit$num_txt
  A <- bandit$txt
  H <- bandit$H

  eps <- bandit$epsilons[t]

  explore <- rbinom(1,1, prob = eps)
  if (explore==1) {
    a <- sample(A, n, replace = T)
    propensity = eps*rep(1,n)/K

    return(list("a" = a,
                "propensity" = propensity))
  } else {
    Vn <- sapply(seq_along(PI), function (i) {
      ipw(pi = PI[[i]],
          X = H[H[,"t"] <= t-1   , varnames_X],
          Y = H[H[,"t"] <= t-1  , "Y"],
          a = H[H[,"t"] <= t-1   , 'a'],
          propensity = H[H[,"t"] <= t-1 , "propensity"])
    })

    idx <- which(Vn == max(Vn))
    pi_opt <- PI[[idx]]
    a <- pi_opt(X)
    propensity = (1-eps)*rep(1,n)

    return(list("a" = a,
                "propensity" = propensity))
  }
}



surrogate_eps_greedy <- function(t, bandit, ncores=1) {

  # Extract bandit related data
  varnames_X <- bandit$varnames_X
  varnames_Z <- bandit$varnames_Z
  p_order <- bandit$p_order
  n <- bandit$n
  K <- bandit$num_txt
  H <- bandit$H
  PI <- bandit$PI

  eps <- bandit$epsilons[t]

  explore <- rbinom(1,1, prob = eps)
  if (explore==1) {
    a <- sample(A, n, replace = T)
    propensity = eps*rep(1,n)/K

    return(list("a" = a,
                "propensity" = propensity))
  } else {

    tm1 <- t-1
    Qs <- do.call(cbind, lapply(seq_along(PI), function (i) {
      q_measure(pi =  PI[[i]],
                X = H[H[,"t"] <=tm1 ,varnames_X],
                t_index = H[H[,"t"] <=tm1, "t"],
                a = H[H[,"t"] <=tm1 , 'a'],
                propensity = H[H[,"t"] <=tm1 , "propensity"])
    }))

    Z <- H[H[ , "t"] <= tm1, varnames_Z]
    R <- relation_matrix(Z, p_order = p_order)
    RL <- relation_list(Z, p_order = p_order)
    constrs <- constraints_sprs(RL, n)


    non_dom_regimes <- non_dominated_class(Qs=Qs, constr = constrs, solver = "glpk", ncores = ncores)

    Vn <- sapply(non_dom_regimes, function (i) {
      ipw(pi = PI[[i]],
          X = H[H[,"t"] <= tm1   , varnames_X],
          Y = H[H[,"t"] <= tm1   , "Y"],
          a = H[H[,"t"] <= tm1   , 'a'],
          propensity = H[H[,"t"] <= tm1 , "propensity"])
    })

    idx <- which(Vn == max(Vn))
    pi_idx <- non_dom_regimes[idx]
    pi_opt <- PI[[pi_idx]]
    a <- pi_opt(X = X)
    propensity = (1-eps)*rep(1,n)

    return(list("a" = a,
                "propensity" = propensity))
  }
}


boot_measure <- function(pi, X, a, propensity) {
    # Boostrapped measure for a given policy
    #  pi: policy function that takes data and returns action
    #  X: covariate history
    #  t_index: time index (decision points) that correspond to data and actions
    #  a: history of actions
    # propensity: history of probability action was assigned

    w <- rexp(n = nrow(X))

    numerator <- w*(a == pi(X))/propensity

    return(numerator/sum(numerator))
  }

