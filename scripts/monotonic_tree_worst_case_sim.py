"""
This file simulates the computation time and memory constraints at the peak
"""
from __future__ import annotations
import pdb

# Local packages
from src.generative_models import CopulaGenerativeModel
from src.strategies import *
from src.helper_functions import uniform_exploration as gen_pix
from src.helper_functions import generate_monotone_linear_spline as gen_gx
from src.monotonic_tree import ConstrainedMonotoneTreeRegressor
from sklearn.linear_model import LinearRegression
import time

# 20 participants * 84 days * 4 decision points per day
n = 20*84*4
p = 20
q = 5
num_actions = 36
theta = list()
theta.append(np.zeros((p, q)))
theta[0][0:4, 0] = np.array([1, 1, 2, 3])
theta[0][4:8, 1] = np.array([-1, 1, 0.5, 0.5])
theta[0][8:12, 2] = np.array([1, 3, -1, 1])

for i in range(num_actions-1):
    theta.append((2*(i+1))*(-1)**(i+1) * theta[0])

ar_rho = 0.3
g = gen_gx()
pi = gen_pix(num_actions)
log_normal_mean = 1.0
log_normal_var = 0.25 ** 2
normal_var = 10
z_var = 1.0

gen_model = CopulaGenerativeModel(
        monotone_func=g,
        log_normal_mean=log_normal_mean,
        log_normal_var=log_normal_var,
        autoregressive_cov=ar_rho,
        context_ndim=p,
        num_actions=num_actions,
        action_selection=pi,
        coefficients=theta,
        normal_var=normal_var,
        surrogate_var=z_var
)


bandit = gen_model.gen_sample(n)

# Parameters for monotonic tree emebedding
n_trees = 100
n_estimators = 5

alphas = 10**np.linspace(3,-2,100)
max_samples = .7
n_estimators = 5
cv = 5

mtr_constrained = ConstrainedMonotoneTreeRegressor(
        n_trees = n_trees,
        alpha = alphas,
        max_samples = max_samples,
        n_estimators = n_estimators,
        partial_order = product_order,
        cv = cv
    )


start = time.time()
mtr_constrained.fit(X=bandit._surrogate, y = bandit._outcome)
print(time.time() - start)
