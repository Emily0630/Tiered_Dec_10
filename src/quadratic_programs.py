"""
Contains functions and classes for running the constrained quadratic programs and constrained quadtratic programs
with slack.
"""

import osqp
from gurobipy import GRB
import gurobipy as gp
import time
from joblib import Parallel, delayed
import numpy as np
from pyparsing import alphas

import scipy
from scipy.sparse import dok_matrix, csr_matrix, csc_matrix
from scipy.special import comb as n_choose_k

from src.orders import product_order

import multiprocessing as mp

from itertools import product
from functools import partial
from typing import Optional, Callable, Union

from sklearn.ensemble import RandomTreesEmbedding

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import cvxopt
from mosek import iparam

cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['mosek'] = {iparam.log: 0}

# Comparing with gurobi for speed

GLOBAL_ENV = gp.Env()
GLOBAL_ENV.setParam("OutputFlag", 0)


def get_constraints(embedding, surrogate_outcome, partial_order):
    """
     Builds constraint matrix based on a partial order over the elements in surrogate_outcome.

    :param embedding: nxd high dimension representation of features
    :param surrogate_outcome: nxq array
    :param partial_order: partial order applied to surrogate_outcome
    :return: constraint vector and constraint matrix to be used in quadratic program
    """

    n, d = embedding.shape
    constr_matrix = dok_matrix((int(n_choose_k(n, 2)), d))
    constr_vec = np.ones(int(n_choose_k(n, 2)))

    surrogate_constraint_counts = np.array([])

    k = 0
    num_no_comparisons = 0
    for i in range(n):
        num_comparisons = 0
        for j in range(i):
            if i == j:
                continue
            # returns 1 if z1 <= zj, 0 if z1 == zj, -1 if zj >= zi
            compare = partial_order(
                surrogate_outcome[i, :], surrogate_outcome[j, :])
            if compare == 0:
                num_no_comparisons += 1
                continue

            # Induced constraints:
            # If compare = 1 then this implies embedding_i \beta^T -  embedding_j \beta^T <= 0
            # If compare = -1 then this implies embedding_j \beta^T -  embedding_i \beta^T <= 0
            constr_matrix[k, :] = compare * (embedding[i, :] - embedding[j, :])
            constr_vec[k] = 0
            k += 1
            num_comparisons += 1

        surrogate_constraint_counts = np.hstack(
            [surrogate_constraint_counts, [i]*num_comparisons])

    constr_matrix = constr_matrix[0:k, :]
    constr_vec = constr_vec[0:k]

    return constr_matrix, constr_vec, surrogate_constraint_counts


def compute_row_constr(i, surrogate_outcome, embedding, partial_order):
    """
    Returns row of contraints for ith element of surrogate_outcome.
    Used in parallel construction of constraint matrix.

    :param i: row of surrogate_outcome
    :param surrogate_outcome: n x q matrix of surrogate outcomes
    :param embedding: n x d high dimensionl embedding of surrogate outcomes
    :param partial_order: Binary function that evaluates if on element is >= than the other.
    :return:
    """
    res = []
    for j in range(i):
        if i == j:
            continue
        # returns 1 if z1 <= zj, 0 if z1 == zj, -1 if zj >= zi
        compare = partial_order(
            surrogate_outcome[i, :], surrogate_outcome[j, :])
        if compare == 0:
            continue

        # Induced constraints:
            # If compare = 1 then this implies embedding_i \beta^T -  embedding_j \beta^T <= 0
            # If compare = -1 then this implies embedding_j \beta^T -  embedding_i \beta^T <= 0
        res.append(compare * (embedding[i, :] - embedding[j, :]))

    if res:
        # number of times ith surrogate outcome was had comparison (used to subset contraints matrix later on)
        num_comparsions = len(res)
        surrogate_constraint_counts = np.ones(num_comparsions)*i
        return scipy.sparse.vstack(res, format="csr"), surrogate_constraint_counts


def get_constraints_mp(embedding, surrogate_outcome, partial_order, cpus=None):
    """
    :param embedding: nxd high dimension representation of features
    :param surrogate_outcome: nxq array
    :param partial_order: partial order applied to surrogate_outcome
    :return: constraint vector and constraint matrix to be used in quadratic program
    """

    if cpus is None:
        cpus = mp.cpu_count() - 1

    n, d = embedding.shape

    f = partial(compute_row_constr, surrogate_outcome=surrogate_outcome,
                embedding=embedding, partial_order=partial_order)
    pool = mp.Pool(cpus)
    res = pool.map(f, range(n))
    pool.close()
    res = list(filter(lambda x: x, res))
    if not res:
        print("No surrogate outcomes are comparable")
        return [None]*3

    constr_matrix = scipy.sparse.vstack([elt[0] for elt in res], format="csr")

    # surrogate indices is a vector that maps each row of the contrain matrix to corresponding surrogate outcome
    surrogate_indices = np.concatenate([elt[1] for elt in res])

    constr_vec = np.zeros(constr_matrix.shape[0])

    return constr_matrix.todok(), constr_vec, surrogate_indices


def fit_qp(
        quadratic_term: Union[csr_matrix, csc_matrix],
        linear_term: Union[csr_matrix, csc_matrix, np.ndarray],
        constr_matrix: dok_matrix,
        constr_vec: np.ndarray,
        alpha: float = 0,
        beta_wghts: Union[csc_matrix, csr_matrix, csr_matrix, None] = None,
        use_slack: bool = False,
        gamma: Optional[float] = None) -> np.ndarray:

    regressor = gp.Model(env=GLOBAL_ENV)
    _, dim = quadratic_term.shape

    if beta_wghts is None:
        beta_wghts = np.eye(dim)

    assert beta_wghts.shape == quadratic_term.shape

    quadratic_term += csr_matrix(alpha * beta_wghts)

    beta = regressor.addMVar(
        dim,
        vtype=GRB.CONTINUOUS,
        name="beta",
        lb=-GRB.INFINITY,
        up=GRB.INFINITY
    )

    constr_matrix = constr_matrix.tocsr()
    linear_term = csr_matrix(linear_term)

    regressor.setObjective(beta @ quadratic_term @ beta +
                           linear_term @ beta, GRB.MINIMIZE)

    if use_slack:
        assert gamma is not None, "Specified slack without specifiying gamma."
        dimC = constr_matrix.shape[0]
        zeros = np.zeros(dimC)
        slack = regressor.addMVar(
            dimC,
            vtype=GRB.CONTINUOUS,
            name="slack")
        regressor.addConstr(constr_matrix @ beta + gamma * slack <= constr_vec)
        regressor.addConstr(slack >= zeros)  # This is not needed as lower bound is already zero
    else:
        regressor.addConstr(constr_matrix @ beta <= constr_vec)

    regressor.optimize()

    return np.array([beta[i].X for i in range(dim)])


def fit_sequential_qp(
        quadratic_term: Union[csr_matrix, csc_matrix],
        linear_term: Union[csr_matrix, csc_matrix, np.ndarray],
        constr_matrix: dok_matrix,
        constr_vec: np.ndarray,
        alpha: float = 0,
        tol: float = 1e-5,
        folds: int = 10,
        beta_wghts: Union[csc_matrix, csr_matrix, csr_matrix, None] = None,
        use_slack: bool = False,
        gamma: Optional[float] = None) -> np.ndarray:

    regressor = gp.Model(env=GLOBAL_ENV)
    _, dim = quadratic_term.shape

    if beta_wghts is None:
        beta_wghts = np.eye(dim)

    assert beta_wghts.shape == quadratic_term.shape

    quadratic_term += csr_matrix(alpha * beta_wghts)

    beta = regressor.addMVar(
        dim,
        vtype=GRB.CONTINUOUS,
        name="beta",
        lb=-GRB.INFINITY
    )

    constr_matrix = constr_matrix.tocsr()
    linear_term = csr_matrix(linear_term)
    regressor.setObjective(beta @ quadratic_term @ beta +
                           linear_term @ beta, GRB.MINIMIZE)

    dimC = constr_matrix.shape[0]
    batch = dimC // folds

    if use_slack:
        assert gamma is not None, "Specified slack without specifiying gamma."

        for j in range(folds):
            current = j*batch
            next = batch*(j+1) - 1
            dimC2 = batch
            if j == (folds - 1):
                next = dimC
                dimC2 = dimC - batch*j

            zeros = np.zeros(dimC2)

            slack = regressor.addMVar(
                dimC2,
                vtype=GRB.CONTINUOUS,
                name="slack")
            regressor.addConstr(
                constr_matrix[current:next, :] @ beta + gamma * slack <= constr_vec[current:next])
            regressor.addConstr(slack >= zeros)
            regressor.optimize()

            # Now we remove constraints with larger slack than the tolerance specified.
            # This is removing constraints not at the boundary of the feasible region and in theory not
            # contributing to the solution.
            num_active_constraints = sum(
                np.array(regressor.getAttr(GRB.Attr.Slack)) < tol)

            # print(f"There are {num_active_constraints} active constraints.\n")
            active_constraints_idx = np.where(
                np.array(regressor.getAttr(GRB.Attr.Slack)) < tol)[0]
            inactive_constraints_idx = np.where(
                np.array(regressor.getAttr(GRB.Attr.Slack)) > tol)[0]

            for inactive in inactive_constraints_idx:
                regressor.remove(regressor.getConstrs()[inactive])

            # remove slack variables
            regressor.remove(regressor.getConstrs()[dimC2:2*dimC2-1])

    else:
        for j in range(folds):
            current = j*batch
            next = batch*(j+1) - 1
            dimC2 = batch
            if j == (folds - 1):
                next = dimC
                dimC2 = dimC - batch*j

            regressor.addConstr(
                constr_matrix[current:next, :] @ beta <= constr_vec[current:next])

            regressor.optimize()
            num_active_constraints = sum(
                np.array(regressor.getAttr(GRB.Attr.Slack)) > tol)

            # Now we remove constraints with larger slack than the tolerance specified.
            # This is removing constraints not at the boundary of the feasible region and in theory not
            # contributing to the solution.

            # print(f"There are {num_active_constraints} active constraints.\n")
            active_constraints_idx = np.where(
                np.array(regressor.getAttr(GRB.Attr.Slack)) < tol)[0]
            inactive_constraints_idx = np.where(
                np.array(regressor.getAttr(GRB.Attr.Slack)) > tol)[0]

            for inactive in inactive_constraints_idx:
                regressor.remove(regressor.getConstrs()[inactive])

    return np.array([beta[i].X for i in range(dim)])


if __name__ == "__main__":
    import numpy as np
    from src.orders import product_order

    from bandit_data import BanditData
    from helper_functions import generate_monotone_linear_spline as gen_gx
    from helper_functions import uniform_exploration as gen_pix

    import numpy.random as nr

    from scipy.stats import norm, expon
    from policy import *
    from generative_models import *
    import time
    from orders import product_order

    from sklearn.ensemble import RandomTreesEmbedding

    import time

    # Set parameters for sampling model for bandit data
    n = 100  # number of observations
    p = 15  # number of covariates (i.e. dimension of context vector)
    q = 3  # number of surrogate outcomes
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

    # Set sampling model for bandit data
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

    # Generate sample
    bandit = gen_model.gen_sample(n)

    # Set number of cpus
    n_jobs = mp.cpu_count() - 1

    # Get RandomForrest Embedding
    RFE = RandomTreesEmbedding()

    embedding = RFE.fit_transform(bandit._surrogate)

    # Get Constraint Matrix
    constr_matrix, constr_vec, surrogate_constr_idx = get_constraints_mp(
        embedding=embedding,
        surrogate_outcome=bandit._surrogate,
        partial_order=product_order,
        cpus=n_jobs)

    # Get Quadratic and Linear Terms
    quadratic_term = embedding.T @ embedding
    linear_term = -2 * embedding.T @ bandit._outcome

    # Fit Quadratic Program
    start = time.time()
    beta = fit_qp(
        quadratic_term=quadratic_term,
        linear_term=linear_term,
        constr_matrix=constr_matrix,
        constr_vec=constr_vec,
        alpha=.5)
    print(f"Standard QP Runtime: {time.time()-start}")

    start = time.time()
    # Fit Sequential Quadratic Program
    beta_seq = fit_sequential_qp(
        quadratic_term=quadratic_term,
        linear_term=linear_term,
        constr_matrix=constr_matrix,
        constr_vec=constr_vec,
        alpha=.5)

    print(f"Sequential QP Runtime: {time.time()-start}")

    # print(beta_seq)
    # print(beta)
    # Printing the l2 norm difference between the the solutions
    print(np.sqrt((beta_seq - beta).T @ (beta_seq - beta)))
    # Max absolute difference
    print(np.max(np.abs(beta_seq - beta)))
