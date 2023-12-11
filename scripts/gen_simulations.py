"""
Compare multiple bandit strategies against each other:
    - monotonic tree embedding on surrogate outcomes
    - standard eps greed or bootstrap thompson sampling on primary outcome
    - random guessing
    - true optimal policy (estimate with monte carlo sampling)

Eventually this will include the policy screening methods
"""

from src.helper_functions import uniform_exploration as gen_pix
from src.helper_functions import generate_monotone_linear_spline as gen_gx
from src.generative_models import CopulaGenerativeModel
from src.policy import LinearBasket
from src.orders import product_order
from src.strategies import *

import numpy as np
import numpy.random as npr
import pandas as pd
import os


# Get TaskID (represents unique simulation id) used to save resultss
taskID=int(os.environ['SLURM_ARRAY_TASK_ID'])

n = 100
p = 15
q = 3
num_actions = 2
theta = list()
theta.append(np.zeros((p, q)))
theta[0][0:4, 0] = np.array([1, 1, 2, 3])
theta[0][4:8, 1] = np.array([-1, 1, 0.5, 0.5])
theta[0][8:12, 2] = np.array([1, 3, -1, 1])
theta.append(abs(theta[0]) * -10 - 10)
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
    surrogate_var=z_var,
    mc_iterations=100
)

## Generate random linear policies
num_policies = 100
basket = LinearBasket.generate_random_basket(
    num_policies=num_policies,
    num_actions=num_actions,
    x_dim=p
)

# Vanilla Epsilon Greedy
eps_greedy = IpwEpsGreedy(policies=basket)

# Parameters for monotonic tree emebedding
n_trees = 100
n_estimators = 5
# alphas  = alphas = np.array([.2,.8, 1, 2, 3, 5])
# cv_folds = 5

alphas = .2
cv_folds = None

actions = np.arange(num_actions)

mtbts = MonotoneTreeBootTS(
    actions=actions,
    n_trees=n_trees,
    n_estimators=n_estimators,
    max_samples=.7,
    alpha=.2
)


# Implement random action
guess = RandomAction(actions=actions)

# Define optimal policy
gen_model.get_optimal_action()
opt_action = np.ones(n)*gen_model.optimal_action

average_outcome = {}
average_outcome[str(eps_greedy)] = []
average_outcome[str(mtbts)] = []
average_outcome[str(guess)] = []
average_outcome["optimal"] = []

## Burn-in
bandit_eps = gen_model.gen_sample(n)
bandit_mtbts = gen_model.gen_sample(n)
bandit_guess = gen_model.gen_sample(n)
bandit_opt = gen_model.gen_sample(n)

## Initialize agents
eps_greedy.update(bandit_eps)
mtbts.update(bandit_mtbts, partial_order=product_order)

num_steps = 50
epsilon = .5*np.log(np.arange(1,num_steps+1))/np.arange(1,num_steps+1)**.75
epsilon[0]=1
for t in range(num_steps):
    print(f"Iteration {t}\n\n")
    # Vanilla Epsilon Greedy
    x = gen_model.get_context(n)
    a, propensity = eps_greedy.pick_action(X=x, epsilon=epsilon[t])
    z = gen_model.get_surrogates(x, action=a)
    y = gen_model.get_outcome(z)
    average_outcome[str(eps_greedy)].append(y.mean())
    bandit_eps.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y, propensitoutcome_new=propensity)

    eps_greedy.update(bandit_eps)

    # Monotone Tree Embedding
    a = np.zeros(n)
    N = len(bandit_mtbts)
    for i in range(n):
        a[i] = mtbts.pick_action(bandit_mtbts._z[N-n+i])

    x = gen_model.get_context(num_samples=n)
    z = gen_model.get_surrogates(x, a)

    y = gen_model.get_outcome(z)
    average_outcome[str(mtbts)].append(y.mean())

    bandit_mtbts = bandit_mtbts.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y)
    mtbts.update(bandit_mtbts, partial_order=product_order)

    # Random guess
    a = guess.pick_action(n)
    z = gen_model.get_surrogates(x, action=a)
    y = gen_model.get_outcome(z)
    average_outcome[str(guess)].append(y.mean())
    bandit_guess = bandit_guess.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y)


    # True Optimal Policy
    z = gen_model.get_surrogates(x, action=opt_action)
    y = gen_model.get_outcome(z)
    average_outcome["optimal"].append(y.mean())
    bandit_opt = bandit_opt.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y)



pd.DataFrame(average_outcome).to_csv(f'../results/simulation_{taskID}.csv', index=False)



