##
## @file generative_model.R
## @brief DTR estimators
## @author Marc Brooks
## @note Written hastily and rather poorly
##



# Sample Surrogates ----
sample_Z <- function(X, theta, sd =1) {
  # X is 1xp array
  # theta is pxq matrix
  q <- ncol(theta)
  X %*% theta +sd* matrix(rnorm(n*q), nrow=1, ncol=q)
}

g_j  <- function(mu, beta_j) {
  L    <- length(beta_j)
  mu_l <- (c(1:L) - 1)/L
  beta_j %*% pmax(mu - mu_l, 0)
}

gen_betas <- function(q,L) {
  apply(replicate(q, runif(L)), MARGIN = 2, function(x) {
    sort(x)/sum(x)
  })
}

gen_g_monotone  <- function(q,L) {
  betas <- gen_betas(q,L)
  g <- list()
  for (j in seq_len(q)) {
    g[[j]] <- function(mu) g_j(mu, beta_j = betas[ ,j])
  }
  return(g)
}

get_outcome <- function(Z, g, delta_mu = 1, delta_sig2 = 5, eta_mu = 0, eta_sig2 = 10) {

  # Z is vector of surrogate outcomes
  # g is a list of monotone functions (corresponding to gen_g_monotone), each corresponding to each component of Z

  delta <- rlnorm(1, meanlog = delta_mu, sdlog = sqrt(delta_sig2))
  eta <- rnorm(1, mean = eta_mu, sd = sqrt(eta_sig2))

  q <- length(Z)
  Summand <- 0
  for (j in seq_len(q)) {
    Summand = Summand + (g[[j]](mu = pnorm(Z[j])) + (q-j))*(Z[j] < 0)*prod(Z[seq_len(j-1)]>=0)
  }
  return(delta*Summand + eta)
}


# get_outcome <- function(Z, betas, delta, eta) {
#   q <- length(Z)
#   Summand <- 0
#   for (j in seq_len(q)) {
#     phi_j <- pnorm(Z[j])
#     beta_j <-  betas[ , j]
#     Summand = Summand + (g_j(mu = phi_j, beta_j = beta_j) + (q-j))*(Z[j] < 0)*prod(Z[seq_len(j-1)]>=0)
#   }
#   return(delta*Summand + eta)
# }
