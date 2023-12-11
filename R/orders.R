##
## @file orders.R
## @brief Functions for partial orders
## @author Marc Brooks
## @note Written hastily and rather poorly
##


library(relations)

relation_matrix <- function(Z, p_order) {
  N <- nrow(Z)
  R <- matrix(numeric(N*N), nrow=N)

  for (i in seq_len(N)) {
    for (j in 1:N) {
      R[i,j] = p_order(Z[i, ], Z[j, ])
    }
  }
  return(R)
}

relation_list <- function(Z, p_order) {
  N <- nrow(Z)
  rel <- lapply(1:N, function(i) {
    lapply(1:N, function(j) {
      if (p_order(Z[i, ], Z[j, ])) return(c(i, j))
    }
    )
  })
  do.call(rbind, list.flatten(rel))
}

cover_matrix <- function(rl) {
  # omit edges to same vertex
  rl <- rl[rl[,1]!=rl[,2], ]

  R <- endorelation(graph = data.frame(rl))
  R <- transitive_reduction(R)

  return(relation_incidence(R))

}

cover_relation_list <- function(rl) {
  # omit edges to same vertex
  rl <- rl[rl[,1]!=rl[,2], ]
  R <- endorelation(graph = data.frame(rl))
  cov_rl <- relation_table(relation_cover(R))
  colnames(cov_rl) <- c("from", "to")
  return(cov_rl)
}

## Partial Ordering
product_order <- function(z1,z2) {
  return(all(z1 <= z2))
}
