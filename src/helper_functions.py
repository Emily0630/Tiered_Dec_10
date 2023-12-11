import pdb
import numpy as np
import numpy.random as nr
import scipy.sparse
from scipy.linalg import sqrtm
from typing import Optional, Callable
import cvxopt
from mosek import iparam
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['mosek'] = {iparam.log: 0}

# Comparing with gurobi for speed
import gurobipy as gp
from gurobipy import GRB
license_file = "/Users/ebl8/Dropbox/Tiered OutcomesFromYinyihong/code/pythonProject1/gurobi.lic"
GLOBAL_ENV = gp.Env()
GLOBAL_ENV.setParam("OutputFlag", 0)


def multivariate_ar_normal(
        num_samples: int,
        rho: float,
        dim: int,
        mean: Optional[np.ndarray] = None
) -> np.ndarray:
    if mean is None:
        mean = np.zeros(dim)
    sigma = np.ones((dim, dim))
    for j in range(dim):
        for i in range(j):
            sigma[j, i] = rho**np.abs(i-j)
            sigma[i, j] = sigma[j, i]
    sigma_sqrt = sqrtm(sigma)
    z_std = nr.randn(num_samples*dim).reshape((num_samples, dim))
    return np.dot(z_std, sigma_sqrt)


def generate_monotone_linear_spline(num_knots: Optional[int] = 7) -> Callable:
    beta_coeffs = np.sort(nr.rand(num_knots))
    beta_coeffs /= np.sum(beta_coeffs)
    knots = np.arange(0, 1.0, 1.0/num_knots)

    def gx(x: np.ndarray) -> float:
        return np.dot(np.maximum(x - knots, 0), beta_coeffs)
    return gx


def uniform_exploration(num_actions: int) -> Callable:
    def pix(x: Optional[np.ndarray] = None):
        return nr.choice(list(range(num_actions)), size=1)[0], 1.0/num_actions
    return pix


def cvxopt_check_is_dominated(
        delta: np.ndarray,
        a_matrix: scipy.sparse.dok_matrix,
        h_matrix: np.ndarray
) -> bool:
    a_matrix_items = a_matrix.items()
    num_items = len(a_matrix_items)
    row_indices = np.array(
        [int(i) for ((i, j), v) in a_matrix_items],
        dtype=int
    )
    col_indices = np.array(
        [int(j) for ((i, j), v) in a_matrix_items],
        dtype=int
    )
    values = np.array(
        [v for ((i, j), v) in a_matrix_items],
        dtype=float
    )
    cvxopt_sparse_G = cvxopt.spmatrix(values, row_indices, col_indices)
    cvxopt_matrix_h = cvxopt.matrix(h_matrix)
    cvxopt_matrix_c = cvxopt.matrix(delta)
    cvxopt_matrix_A = cvxopt.matrix(
        np.ones(len(delta)).reshape((1, len(delta)))
    )
    cvxopt_matrix_b = cvxopt.matrix(np.array([1]).reshape((1, 1)), tc='d')

    cvxopt_min = cvxopt.solvers.lp(
        c=cvxopt_matrix_c,
        G=cvxopt_sparse_G,
        h=cvxopt_matrix_h,
        A=cvxopt_matrix_A,
        b=cvxopt_matrix_b,
        solver="mosek"
    )
    if cvxopt_min['status'] != 'optimal':
        print("Opt failed")
        return False
    cvxopt_min_value = np.dot(
        np.array(cvxopt_min['x']).reshape(len(delta),),
        delta
    )
    cvxopt_matrix_c_neg = cvxopt.matrix(-delta)
    cvxopt_max = cvxopt.solvers.lp(
        c=cvxopt_matrix_c_neg,
        G=cvxopt_sparse_G,
        h=cvxopt_matrix_h,
        A=cvxopt_matrix_A,
        b=cvxopt_matrix_b,
        solver="mosek"
    )
    if cvxopt_max['status'] != 'optimal':
        print("Opt failed")
        return False
    cvxopt_max_value = np.dot(
        np.array(cvxopt_max['x']).reshape(len(delta), ),
        delta
    )
    return cvxopt_min_value >= -1e-4 and cvxopt_max_value > 1e-3


def gurobi_check_is_dominated(
        delta: np.ndarray,
        a_matrix: scipy.sparse.dok_matrix,
        h_matrix: np.ndarray,
        s_matrix: scipy.sparse.dok_matrix = None,
        s_vector: np.ndarray = None
) -> bool:
    min_model = gp.Model("min_model", env=GLOBAL_ENV)
    x = min_model.addMVar(
        shape=len(delta),
        vtype=GRB.CONTINUOUS,
        name="x",
        lb=np.ones(len(delta))/len(delta),
        ub=np.ones(len(delta))
    )
    min_model.setObjective(delta @ x, GRB.MINIMIZE)
    min_model.addConstr(a_matrix @ x <= h_matrix)
    min_model.addConstr(np.ones(len(delta)) @ x >= 1)
    if s_matrix is not None:
        min_model.addConstr(s_matrix @ x <= s_vector)
    min_model.optimize()
    min_model_value = min_model.ObjVal

    #max_model = min_model.copy()
    #max_model.setObjective(delta @ x, GRB.MAXIMIZE)
    #max_model.optimize()
    #max_model_value = max_model.ObjVal
    return min_model_value >= -1e-4 #and max_model_value >= 1e-3


