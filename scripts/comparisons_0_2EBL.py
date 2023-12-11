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
from src.generative_models import CopulaGenerativeModel, TestGenerativeModel, TestDeuxGenerativeModel
from src.policy import LinearBasket
from src.orders import product_order, no_order
from src.strategies import *
import datetime

import numpy as np
import numpy.random as npr
import pandas as pd
import os
import time
import pdb

print(os.getcwd())
output_path = \
    r'C:\Users\laber\Dropbox\Tiered OutcomesFromYinyihong\code\pythonProject1\results_no_MT\\'
n = 20
p = 10
q = 3
num_actions = 2
theta = list()
theta.append(5*np.ones((p, q))/np.sqrt(p))
theta.append(-5*np.ones((p, q))/np.sqrt(p))

ar_rho = 0.3
g = gen_gx()
pi = gen_pix(num_actions)
log_normal_mean = 1.0
log_normal_var = 0.5 ** 2
normal_var = 10
z_var = 0.10

gen_model = TestDeuxGenerativeModel(
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
num_policies = 100
basket = LinearBasket.generate_random_basket(
    num_policies=num_policies,
    num_actions=num_actions,
    x_dim=p
)

# Vanilla Epsilon Greedy
eps_greedy = IpwEpsGreedy(policies=basket)

# Policy Screening Epsilon Greedy
ps_eps_greedy = PolicyScreeningEpsGreedy(policies=basket)

# Vanilla Boot TS
boot_ts = IpwBootTS(policies=basket)

# Policy Screening Boot TS
ps_boot_ts = PolicyScreeningBootTS(policies=basket)

# Parameters for monotonic tree emebedding
n_trees = 100
n_estimators = 5
# alphas  = alphas = np.array([.2,.8, 1, 2, 3, 5])
# cv_folds = 5

alphas = .2
cv_folds = None

actions = np.arange(num_actions)


# Implement random action
guess = RandomAction(actions=actions)

# Define optimal policy
gen_model.get_optimal_action()
opt_action = np.ones(n) * gen_model.optimal_action

#average_outcome = {str(eps_greedy): [], str(ps_eps_greedy): [], str(boot_ts): [], str(ps_boot_ts): [], str(mtbts): [], str(mtbeg): [], str(guess): [], "optimal": []}
sum_outcome = {str(eps_greedy): [], str(ps_eps_greedy): [], str(boot_ts): [], str(ps_boot_ts): [], str(guess): [], "optimal": []}

## Burn-in
bandit_eps = gen_model.gen_sample(n)
bandit_ps_eps = gen_model.gen_sample(n)
bandit_ts = gen_model.gen_sample(n)
bandit_ps_ts = gen_model.gen_sample(n)
bandit_mtbeg = gen_model.gen_sample(n)
bandit_mtbts = gen_model.gen_sample(n)
bandit_mtbtsnc = gen_model.gen_sample(n)
bandit_guess = gen_model.gen_sample(n)
bandit_opt = gen_model.gen_sample(n)

## Initialize agents
eps_greedy.update(bandit_eps)
ps_eps_greedy.update(bandit_ps_eps, partial_order=product_order)
boot_ts.update(bandit_ts)
ps_boot_ts.update(bandit_ps_ts, partial_order=product_order)

num_steps = 100
epsilon = 0.5 * np.log(np.arange(1, num_steps + 1)) / np.arange(1, num_steps + 1) ** .75
epsilon[0] = 1
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
    bandit_eps = bandit_eps.add_observations(
        context_new=x,
        action_new=a,
        surrogate_new=z,
        outcome_new=y,
        propensity_new=propensity
    )

    eps_greedy.update(bandit_eps)
    print(f"Epsilon Greedy finish, Running Time: {time.time() - t1:.2f}")
    t1 = time.time()

    # Policy Screening  Epsilon Greedy
    x = gen_model.get_context(n)
    a, propensity = ps_eps_greedy.pick_action(x, epsilon=epsilon[t])
    z = gen_model.get_surrogates(x, a)
    y = gen_model.get_outcome(z)
    #average_outcome[str(ps_eps_greedy)].append(y.mean())
    sum_outcome[str(ps_eps_greedy)].append(y.sum())
    bandit_ps_eps = bandit_ps_eps.add_observations(
        context_new=x,
        action_new=a,
        surrogate_new=z,
        outcome_new=y,
        propensity_new=propensity
    )

    ps_eps_greedy.update(bandit_ps_eps, partial_order=product_order)
    print(f"Policy Screening Epsilon Greedy finish, Running Time: {time.time() - t1:.2f}")
    t1 = time.time()

    # Vanilla Boot TS
    x = gen_model.get_context(n)
    a, propensity = boot_ts.pick_action(context=x)
    z = gen_model.get_surrogates(x, a)
    y = gen_model.get_outcome(z)
    #average_outcome[str(boot_ts)].append(y.mean())
    sum_outcome[str(boot_ts)].append(y.sum())
    bandit_ts = bandit_ts.add_observations(
        context_new=x,
        action_new=a,
        surrogate_new=z,
        outcome_new=y,
        propensity_new=propensity
    )

    boot_ts.update(bandit_ts)
    print(f"Vanilla Boot TS finish, Running Time: {time.time() - t1:.2f}")
    t1 = time.time()

    # Policy Screening Boot TS
    x = gen_model.get_context(n)
    a, propensity = ps_boot_ts.pick_action(x, epsilon=0.0)
    z = gen_model.get_surrogates(x, a)
    y = gen_model.get_outcome(z)
    #average_outcome[str(ps_boot_ts)].append(y.mean())
    sum_outcome[str(ps_boot_ts)].append(y.sum())
    bandit_ps_ts = bandit_ps_ts.add_observations(
        context_new=x,
        action_new=a,
        surrogate_new=z,
        outcome_new=y,
        propensity_new=propensity
    )

    ps_boot_ts.update(bandit_ps_ts, partial_order=product_order)
    print(f"Policy Screening Boot TS finish, Running Time: {time.time() - t1:.2f}")
    t1 = time.time()

    # Random guess
    a = guess.pick_action(n)
    z = gen_model.get_surrogates(x, action=a)
    y = gen_model.get_outcome(z)
    #average_outcome[str(guess)].append(y.mean())
    sum_outcome[str(guess)].append(y.sum())
    bandit_guess = bandit_guess.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y)
    print(f"Random Guess finish, Running Time: {time.time() - t1:.2f}")
    t1 = time.time()

    # True Optimal Policy
    z = gen_model.get_surrogates(x, action=opt_action)
    y = gen_model.get_outcome(z)
    #average_outcome["optimal"].append(y.mean())
    sum_outcome["optimal"].append(y.sum())
    bandit_opt = bandit_opt.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y)
    print(f"True Optimal finish, Running Time: {time.time() - t1:.2f}")
    t1 = time.time()

    print(sum_outcome)
 
    cur_time = datetime.datetime.now().replace(microsecond=0)
    #pdb.set_trace()
    pd.DataFrame(sum_outcome).to_csv(
        "../results_no_MT/" + f'comparison_0_1_{t}_{num_steps}.csv',
        index=False
    )
    print(
        pd.DataFrame(sum_outcome).sum(axis=0)
    )
    #print(f"Writing result finish, Running Time: {time.time() - t1:.2f}")

