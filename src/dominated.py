from bandit_data import BanditData
from policy import LinearPolicy, LinearBasket
import numpy as np
import numpy.random as nr
import scipy
from scipy.sparse import dok_matrix, csr_matrix, csc_matrix
from scipy.special import comb as n_choose_k
import gurobipy as gp
from gurobipy import GRB

GLOBAL_ENV = gp.Env()
GLOBAL_ENV.setParam("OutputFlag", 0)


def check_is_dominated(q_to_check: np.ndarray, q_set: np.ndarray, data: BanditData) -> bool:
    n = len(data)
    k = bandit_data.get_surrogate_dim()
    linear_constraint_matrix = dok_matrix((int(n_choose_k(n, 2)), k))
    linear_constraint_rhs = np.zeros(int(n_choose_k(n, 2)))
    counter = 0
    for i in range(n):
        for j in range(i):
            comparison = data.i_less_j(i, j)
            if comparison == 0:
                continue
            if comparison == 1:
                linear_constraint_matrix[counter, i] = 1
                linear_constraint_matrix[counter, j] = -1
                linear_constraint_rhs[counter] = 0
            else:
                linear_constraint_matrix[counter, i] = -1
                linear_constraint_matrix[counter, j] = 1
                linear_constraint_rhs[counter] = 0
            counter += 1
    linear_constraint_matrix = linear_constraint_matrix[:counter, :]
    linear_constraint_rhs = linear_constraint_rhs[:counter]

    # Setup the Gurobi model
    model = gp.Model("lp_model", env=GLOBAL_ENV)
    num_q = q_set.shape[0]

    # weights representing the latent mean at points i=1,2,...,n we called
    # this a_i in our write-up (but not using 'a' to avoid conflating with
    # action etc.
    weights = model.addVars(
        n,
        vtype=GRB.CONTINUOUS,
        name="weights",
        lb=0.0,
        ub=1.0
    )

    mean_diffs = model.addVars(
        num_q,
        vtype=GRB.CONTINUOUS,
        name="mean_diffs",
        lb=-GRB.INFINITY
    )

    # Add the linear constraints
    for i in range(linear_constraint_matrix.shape[0]):
        model.addConstr(
            gp.quicksum(
                linear_constraint_matrix[i, j] * weights[j]
                for j in range(n)
            ) <= linear_constraint_rhs[i]
        )


    # variable coding objective
    W = model.addVar(lb=-GRB.INFINITY, name="W")
    for j in range(num_q):
        model.addConstr(
            W <= gp.quicksum(
                weights[i] * q_set[j, i] - weights[i]*q_to_check[i]
                for i in range(n)
            )
        )

    model.setObjective(W, GRB.MAXIMIZE)
    model.optimize()
    return model.ObjVal <= 0


if __name__ == "__main__":
    n = 100
    p = 10
    k = 3 # number of surrogates
    x = nr.normal(size=(n, p))
    a = np.sign(nr.randn(n))
    b = nr.normal(size=(p, k))
    z = x @ b
    y = a*(z @ np.ones(k)) + nr.normal(size=n)
    propensity = np.ones(n)/2.0
    bandit_data = BanditData()
    bandit_data = bandit_data.add_observations(x, a, z, y, propensity)


    # Create linear policy
    beta = [np.ones(p), -np.ones(p)]
    linear_policy = LinearPolicy(num_actions=2, beta=beta)
    basket = LinearBasket()
    basket = basket.generate_random_basket(100, 2, p)


