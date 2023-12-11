from __future__ import annotations
import pdb

# Local packages
from src.generative_models import CopulaGenerativeModel
from src.strategies import *
from src.helper_functions import uniform_exploration as gen_pix
from src.helper_functions import generate_monotone_linear_spline as gen_gx
from sklearn.linear_model import LinearRegression

n = 100
p = 15
q = 3
num_txts = 2
theta = list()
theta.append(np.zeros((p, q)))
theta[0][0:4, 0] = np.array([1, 1, 2, 3])
theta[0][4:8, 1] = np.array([-1, 1, 0.5, 0.5])
theta[0][8:12, 2] = np.array([1, 3, -1, 1])
theta.append(-theta[0])
ar_rho = 0.3
g = gen_gx()
pi = gen_pix(num_txts)
log_normal_mean = 1.0
log_normal_var = 0.25 ** 2
normal_var = 10
z_var = 1.0
num_actions = 2

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
# alphas  = alphas = np.array([.2,.8, 1, 2, 3, 5])
# cv_folds = 5

alphas = 2
cv_folds = None

txts = np.arange(num_txts)

agent = MonotoneTreeBootTS(
    actions=txts,
    n_trees=n_trees,
    n_estimators=n_estimators,
    max_samples=.7,
    model=LinearRegression,
    alpha=.2
)


def unwrap_less_than(z1, z2):
    return bandit._less_than(z1, z2)


import time

T = 50
start = time.time()
for t in range(T):
    print("Iteration {}".format(t))


    startt = time.time()
    agent.update(bandit, partial_order=unwrap_less_than)
    print((time.time() - startt) / 60)

    # select actions
    

    x = gen_model.get_context(num_samples=n)
    a = agent.pick_action(context=x)
    z = gen_model.get_surrogates(x, a)

    y = gen_model.get_outcome(z)

    bandit = bandit.add_observations(
        context_new=x,
        action_new=a,
        surrogate_new=z,
        outcome_new=y
    )

print(time.time() - start)
