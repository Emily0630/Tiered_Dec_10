"""
Comparing monotone tree regression with constrains to monotone tree regression without restraints to standard linear model.
"""
# Imports 
from curses import reset_shell_mode
from statistics import mean
import numpy as np
from src.monotonic_tree import MonotoneTreeRegressor, ConstrainedMonotoneTreeRegressor
from src.orders import product_order

from src.bandit_data import BanditData
from src.helper_functions import generate_monotone_linear_spline as gen_gx
from src.helper_functions import uniform_exploration as gen_pix

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

from src.generative_models import *

import multiprocessing as mp

import argparse
import os 

parser=argparse.ArgumentParser()
parser.add_argument('--n_samples', help='Number of observations', type=int)
parser.add_argument('--n_trees', help='Number of trees to use in random forest embedding', type=int)
parser.add_argument('--n_actions', help='Number of available actions.', type=int)

parser.add_argument('--p', help='Dimension of context (feature) space', type=int)
parser.add_argument('--q', help='Dimension of surrogate outcome space', type=int)
parser.add_argument('--iteration', help='Iteration number', type=int)
parser.add_argument('--dir_path', help='Path to save results', type=str)



print(os.getcwd() + '\n')

# Extract arguments
args=parser.parse_args()

n_samples = args.n_samples
n_trees = args.n_trees
p = args.p
q = args.q
iteration = args.iteration
num_actions = args.n_actions
dir_path = args.dir_path


print('Initializing generative model\n')
theta = list()
theta.append(np.zeros((p, q)))
theta[0][0:4, 0] = np.array([1, 1, 2, 3])
theta[0][4:8, 1] = np.array([-1, 1, 0.5, 0.5])
theta[0][8:12, 2] = np.array([1, 3, -1, 1])
theta.append(-theta[0])
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

# Testing classes for estimators
# alphas = 10**np.linspace(3,-2,8)
alphas = [0.01, 1, 10]
# alphas = 1
max_samples = .6
n_estimators = 5
cv = 5

mtr_params = dict(n_trees = n_trees, alpha = alphas)

cmtr_params =  dict(
        n_trees = n_trees,
        alpha = alphas,
        max_samples = max_samples,
        n_estimators = n_estimators,
        partial_order = product_order,
        n_jobs = mp.cpu_count() - 2,
        cv = cv
    )

print("Running models\n")
mtr = MonotoneTreeRegressor(**mtr_params)
cmtr = ConstrainedMonotoneTreeRegressor(**cmtr_params)
lr = RidgeCV(alphas=alphas)

train_set = gen_model.gen_sample(n_samples)
test_set  = gen_model.gen_sample(n_samples)

mtr.fit(X=train_set._surrogate, y = train_set._outcome)
cmtr.fit(X=train_set._surrogate, y = train_set._outcome)
_ = lr.fit(X=train_set._surrogate, y = train_set._outcome)

y_mtr  = mtr.predict(X=test_set._surrogate)
y_cmtr = cmtr.predict(X=test_set._surrogate)
y_lr   = lr.predict(X=test_set._surrogate)

mtr_rmse  = np.sqrt(mean_squared_error(test_set._outcome, y_mtr))
cmtr_rmse = np.sqrt(mean_squared_error(test_set._outcome, y_cmtr))
lr_rmse   = np.sqrt(mean_squared_error(test_set._outcome, y_lr))

file_path = os.path.join(dir_path, str(iteration) + '.csv')
print(f'Writing results to {file_path}\n')

with open(file_path, 'w') as f:
    f.write(f"{mtr_rmse},{cmtr_rmse},{lr_rmse}")

