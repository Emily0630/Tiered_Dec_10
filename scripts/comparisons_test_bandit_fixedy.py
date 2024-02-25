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

n = 50
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
num_policies = 100
basket = LinearBasket.generate_random_basket(
    num_policies=num_policies,
    num_actions=num_actions,
    x_dim=p
)

actions = np.arange(num_actions)


# Vanilla Epsilon Greedy
eps_greedy = IpwEpsGreedy(policies=basket)

guess = RandomAction(actions=actions)

# Vanilla Boot TS
boot_ts = IpwBootTS(policies=basket, replicates=2)

alphas = .2
cv_folds = None




# sum_outcome = {str(eps_greedy): [], str(boot_ts): []}
sum_outcome = {str(eps_greedy): [], str(boot_ts): [], 'RandomAction': [], 'eps_greedy_policy': [],
               "optimal_outcome_actions":[]}
## Burn-in
bandit_eps = gen_model.gen_sample(n)
bandit_guess = gen_model.gen_sample(n)
bandit_ts = gen_model.gen_sample(n)
# print(f"Optimal policy: {eps_greedy._policies._basket.index(eps_greedy.optimal_policy)}")
## Initialize agents
eps_greedy.update(bandit_eps)
boot_ts.update(bandit_ts)

print(f"Optimal policy: {eps_greedy._policies._basket.index(eps_greedy.optimal_policy)}")
print("finish initialization")
num_steps = 200
epsilon = .5 * np.log(np.arange(1, num_steps + 1)) / np.arange(1, num_steps + 1) ** .75
epsilon[0] = 1
t0 = time.time()
for t in range(num_steps):
    
    t1 = time.time()
    # Vanilla Epsilon Greedy
    x = gen_model.get_context(n)
    a, propensity = eps_greedy.pick_action(context=x, epsilon=epsilon[t])
    z = gen_model.get_surrogates(x, a)
    y = gen_model.get_outcome(z)
    sum_outcome[str(eps_greedy)].append(y.sum())
    sum_outcome["eps_greedy_policy"].append(eps_greedy._policies._basket.index(eps_greedy.optimal_policy))
    bandit_eps = bandit_eps.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y, propensity_new=propensity)

    eps_greedy.update(bandit_eps)
    print(f"Epsilon Greedy finish, Running Time: {time.time() - t1:.2f}")
    
    t1 = time.time()
   # Vanilla Boot TS
    x = gen_model.get_context(n)
    a, propensity = boot_ts.pick_action(context=x)
    z = gen_model.get_surrogates(x, a)
    y = gen_model.get_outcome(z)
    sum_outcome[str(boot_ts)].append(y.sum())
    bandit_ts = bandit_ts.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y, propensity_new=propensity)

    boot_ts.update(bandit_ts)
    print(f"Vanilla Boot TS finish, Running Time: {time.time() - t1:.2f}")
    t1 = time.time()
    
    # Random guess
    a = guess.pick_action(n)
    z = gen_model.get_surrogates(x, action=a)
    y = gen_model.get_outcome(z)
    #average_outcome[str(guess)].append(y.mean())
    sum_outcome[str(guess)].append(y.sum())
    bandit_guess = bandit_guess.add_observations(context_new=x, action_new=a, surrogate_new=z, outcome_new=y)
    t1 = time.time()
    
    # breakpoint()
    
    sum_outcome["optimal_outcome_actions"].append(gen_model.get_outcome_optimal_actions(x).sum())
    # sum_outcome["optimal_policy"].append(gen_model.get_optimal_policy(bandit_eps, basket))
    # sum_outcome["optimal_outcome_policy"].append(gen_model.get_outcome_optimal_policy(x).sum())

    if t % 10 == 0:
        print(f"\n\nIteration {t}, Current Running Time: {time.time() - t0:.2f}")
        # print(f"Epsilon Greedy finish, Running Time: {time.time() - t1:.2f}")
        print(bandit_eps._context.shape)
        print(sum_outcome)
        pd.DataFrame(sum_outcome).to_csv(f'results_temp/comparison_fixedy_{num_steps}.csv', index=False)
        # breakpoint()
        print(f"Optimal policy: {eps_greedy._policies._basket.index(eps_greedy.optimal_policy)}")

