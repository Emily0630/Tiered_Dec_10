"""
Compare multiple bandit strategies against each other:
    - monotonic tree embedding using thompson sampling on surrogate outcomes
    - monotonic tree embedding using epsilon greedy on surrogate outcomes
    - monotonic tree embedding using thompson sampling on surrogate outcomes without monotone constraints
    - standard eps greed or bootstrap thompson sampling on primary outcome
    - policy screening eps greedy or bootstrap thompson sampling
    - random guessing
    - true optimal policy (estimate with monte carlo sampling)
"""

from src.helper_functions import uniform_exploration as gen_pix
from src.helper_functions import generate_monotone_linear_spline as gen_gx
from src.generative_models import CopulaGenerativeModel
from src.policy import LinearBasket
from src.orders import product_order, no_order
from src.strategies import *
import datetime;

import numpy as np
import numpy.random as npr
import pandas as pd

import time

n = 5
p = 25
q = 10
num_actions = 2
theta = list()
theta.append(np.zeros((p, q)))
theta[0][0:4, 0] = np.array([1, 1, 2, 3])
theta[0][4:8, 1] = np.array([-1, 1, 0.5, 0.5])
theta[0][8:12, 2] = np.array([1, 3, -1, 1])
theta.append(abs(theta[0]) * -10 - 10)
ar_rho = 0.5
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
    mc_iterations=10000
)

## Generate random linear policies
num_policies = 20
basket = LinearBasket.generate_random_basket(
    num_policies=num_policies,
    num_actions=num_actions,
    x_dim=p
)

# Vanilla Epsilon Greedy
eps_greedy = IpwEpsGreedy(policies=basket)

# Policy Screening Epsilon Greedy
ps_eps_greedy = PolicyScreeningEpsGreedy(policies=basket)




alphas = .2
cv_folds = None

actions = np.arange(num_actions)


sum_outcome = {str(eps_greedy): [], str(ps_eps_greedy): [], "non_dominated_policies": []}
## Burn-in
bandit_eps = gen_model.gen_sample(n)
bandit_ps_eps = gen_model.gen_sample(n)

## Initialize agents
eps_greedy.update(bandit_eps)
ps_eps_greedy.update(bandit_ps_eps, partial_order=product_order)

num_steps = 50
epsilon = .5 * np.log(np.arange(1, num_steps + 1)) / np.arange(1, num_steps + 1) ** .75
epsilon[0] = 1
epsilon *= 0
t0 = time.time()
for t in range(num_steps):
    print(f"\n\nIteration {t}, Current Running Time: {time.time() - t0:.2f}")
    t1 = time.time()
    # Vanilla Epsilon Greedy
    x = gen_model.get_context(n)
    a, propensity = eps_greedy.pick_action(context=x, epsilon=epsilon[t])
    z = gen_model.get_surrogates(x, a)
    y = gen_model.get_outcome(z)
    #average_outcome[str(eps_greedy)].append(y.mean())
    sum_outcome[str(eps_greedy)].append(y.sum())
    bandit_eps = bandit_eps.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y, propensity_new=propensity)

    eps_greedy.update(bandit_eps)
    print(f"Epsilon Greedy finish, Running Time: {time.time() - t1:.2f}")
    t1 = time.time()

    # Policy Screening  Epsilon Greedy
    x = gen_model.get_context(n)
    a, propensity = ps_eps_greedy.pick_action(x, epsilon=epsilon[t])
    z = gen_model.get_surrogates(x, a)
    y = gen_model.get_outcome(z)
    # average_outcome[str(ps_eps_greedy)].append(y.mean())
    sum_outcome[str(ps_eps_greedy)].append(y.sum())
    # breakpoint()
    bandit_ps_eps = bandit_ps_eps.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y, propensity_new=propensity)

    ps_eps_greedy.update(bandit_ps_eps, partial_order=product_order)
    print(f"Policy Screening Epsilon Greedy finish, Running Time: {time.time() - t1:.2f}")
    sum_outcome["non_dominated_policies"].append(ps_eps_greedy._non_dominated_indices)
    t1 = time.time()


    print(sum_outcome)
 
    cur_time = datetime.datetime.now().replace(microsecond=0)

    #pd.DataFrame(sum_outcome).to_csv(f'results2/comparison_{cur_time}_{t}_{num_steps}.csv', index=False)
    pd.DataFrame(sum_outcome).to_csv(f'results_Mar21/comparison_{t}_{num_steps}.csv', index=False)
    print(f"Writing result finish, Running Time: {time.time() - t1:.2f}")

