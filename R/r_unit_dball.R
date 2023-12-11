##
## @file r_unit_dball.R
## @brief
## @author Marc Brooks
## @note Written hastily and rather poorly
##

# Sample from Unit ball -----
r_unit_dball <- function(d, n) {
  S <- matrix(rnorm(d*n), ncol = d)
  U_d <- runif(n)^(1/d)

  U_d *(S / apply(S, MARGIN = 1, function(x) sqrt(sum(x^2))))
}

# Tests
s3 <- r_unit_dball(d=3, n=1000)
plot3d(x = s3[,1], y = s3[,2], z = s3[,3])
rglwidget()
