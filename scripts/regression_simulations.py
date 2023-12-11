"""
Comparing monotone tree regression with constrains to monotone tree regression without restraints to standard linear model.
"""
# Imports 
from curses import reset_shell_mode
from statistics import mean
import numpy as np
import pandas as pd
from src.monotonic_tree import MonotoneTreeRegressor, ConstrainedMonotoneTreeRegressor
from src.orders import product_order

from src.bandit_data import BanditData
from src.helper_functions import generate_monotone_linear_spline as gen_gx
from src.helper_functions import uniform_exploration as gen_pix

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

from src.generative_models import *

import multiprocessing as mp
from joblib import Parallel, delayed

import itertools
from functools import partial
import time

n = 100
p = 15
q = 3
num_actions = 2
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
# Monotone embedding 
n_trees = 100
n_estimators = 10
# alphas = 10**np.linspace(3,-2,8)
alphas = [0.01, 1, 10]
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


# Function for parallel computation
def regression_compare_par(gen_model, alphas, mtr_params, cmtr_params, n_samples):
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
    
    return mtr_rmse, cmtr_rmse, lr_rmse


# Simulation params
cpus = mp.cpu_count()-2
niter = 500
params_list = [dict(n_samples = params[0], n_trees = params[1]) \
    for params in itertools.product([100, 250], [100, 200, 500])]
df_index=pd.MultiIndex.from_product([[str(p) for p in params_list],['mean', 'sd']])

results = pd.DataFrame(np.zeros((2*len(params_list), 3)), index=df_index, columns=['MTR', 'CMTR', 'RLR'])

for i, params in enumerate(params_list):

    print("-"*10 + " Parameters: n_samples = {n_samples}, n_trees = {n_trees} ".format(**params)+ "-"*10 + "\n")

    mtr_params['n_trees']  = params['n_trees']
    cmtr_params['n_trees'] = params['n_trees']

    estimates = Parallel(
        n_jobs=cpus,
        prefer="threads",
    )(delayed(regression_compare_par)(
        gen_model=gen_model, 
        alphas=alphas,
        mtr_params=mtr_params,
        cmtr_params=cmtr_params,
        n_samples=params['n_samples']
    )
        for i in range(niter)
    )
    
    mtr_rmse_mean = np.mean([elt[0] for elt in estimates])
    cmtr_rmse_mean = np.mean([elt[1] for elt in estimates])
    lr_rmse_mean = np.mean([elt[2] for elt in estimates])

    mtr_rmse_sd = np.std([elt[0] for elt in estimates])
    cmtr_rmse_sd = np.std([elt[1] for elt in estimates])
    lr_rmse_sd = np.std([elt[2] for elt in estimates])


    results.loc[(str(params), 'mean')] = [mtr_rmse_mean, cmtr_rmse_mean, lr_rmse_mean]
    results.loc[(str(params), 'sd')]   = [mtr_rmse_sd, cmtr_rmse_sd, lr_rmse_sd]

results.to_csv('./results/regression_sim.csv', index=False)
