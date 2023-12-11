import itertools
import time
from joblib import Parallel, delayed
import numpy as np
from pyparsing import alphas

import scipy
from scipy.sparse import dok_matrix, csr_matrix, csc_matrix
from scipy.special import comb as n_choose_k

from src.orders import product_order

from src.quadratic_programs import *

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
import gurobipy as gp
from gurobipy import GRB
license_file = "/Users/ebl8/Dropbox/Tiered OutcomesFromYinyihong/code/pythonProject1/gurobi.lic"
GLOBAL_ENV = gp.Env()
GLOBAL_ENV.setParam("OutputFlag", 0)




def fit_monotone_tree(
        surrogate_outcome: np.ndarray,
        embedding: np.ndarray,
        outcome: np.ndarray,
        partial_order: Callable,
        constr_matrix: Optional[scipy.sparse.dok_matrix] = None,
        constr_vec: Optional[np.ndarray] = None,
        alpha: Optional[int] = None,
        beta_wghts: Union[np.ndarray, csr_matrix, csc_matrix] = None,
        wghts: Optional[np.ndarray] = None,
        use_slack: bool = False,
        gamma: Optional[float] = None
) -> Callable:
    """Estimator for primary outcome using high dimensional embedding of surrogate outcomes as predictors and adaptive ridge.

    Args:
        surrogate_outcome (np.ndarray): Vector of outcomes interpreted as surrogates for primary outcome. Here they are used to predict the primary outcome.
        embedding (np.ndarray): High dimensional embedding of surrogate outcomes.
        outcome (np.ndarray): Primary outcome variable we are trying to estimate.
        partial_order (Callable): function that represents partial order used to generate constaint matrix.
        constr_matrix (Optional[scipy.sparse.dok_matrix], optional): Constraint matrix for quadratic program. These will be the constraints induced
                                                                     by the surrogate outcomes.Defaults to None.
        constr_vec (Optional[np.ndarray], optional): Right hand side of constraints inputed into quadratic program. Defaults to None.
        alpha (Optional[int], optional): hyperparameter that controls amount of regularization. Defaults to None.
        beta_wghts (Optional[np.ndarray], optional): weights to adapt ridge penalty (i.e. adaptive ridge).
        wghts (Optional[np.ndarray], optional): weights for scaling data.
        use_slack (bool): Indicates whether or not to use slack for soft constraints. Defaults to False..
        gamma (Optional[np.ndarray], optional): Tuning paramter for how much slack to use.

    Returns:
        Callable: _description_
    """
    if alpha is None:
        alpha = 0

    if wghts is None:
        wghts = np.ones_like(outcome)

    if (constr_vec is None) and (constr_matrix is None):
        # constr_matrix, constr_vec, surrogate_constr_idx = get_constraints_mp(embedding=embedding, surrogate_outcome=surrogate_outcome, partial_order=partial_order)
        constr_matrix, constr_vec, surrogate_constr_idx = get_constraints(embedding=embedding, surrogate_outcome=surrogate_outcome, partial_order=partial_order)

    W = csr_matrix(np.diag(wghts))

    quadratic_term = embedding.T @ W @ embedding
    linear_term = -2 * embedding.T @ (W @ outcome)

    beta = fit_sequential_qp(quadratic_term=quadratic_term, linear_term=linear_term, \
        constr_matrix=constr_matrix, constr_vec=constr_vec, alpha=alpha, \
        beta_wghts = beta_wghts, use_slack = use_slack, gamma = gamma)

    def predict(embedding):
        dim = embedding.shape
        p = dim[0]
        if len(dim) == 2:
            p = dim[1]
        assert p == beta.shape[0], f"Expected array with dimension {beta.shape[0]}"

        return embedding @ beta

    return predict

def _scale_data(
    X: Union[np.ndarray, csr_matrix],
    y: np.ndarray,
    w: np.ndarray 
      ):
    """Scales data with weights. Used in bootstrap thompson sampling

    Args:
        X (Union[np.ndarray, scipy.sparse.csr_matrix]): _description_
        y (np.ndarray): _description_

    Returns:
        Tuple(scipy.sparse.csr_matrix, np.np.ndarray): tuple of scaled X and y
    """

    if w is None:
        w = np.ones_like(y)
        w = np.sqrt(w)
        
    
    W = csr_matrix(np.diag(w))

    new_X = W.dot(X)
    new_y = W.dot(y)

    return new_X, new_y

    

def _get_sample_indices(
        n_samples: int,
        max_samples: Union[int, float, None],
        seed: Optional[int] = None
) -> np.ndarray:
    """Private function used to generate indices used in subsampling and aggregation.

    Args:
        n_samples (int): Number of samples we are taking subsample from.
        max_samples (Union[int, float, None]): Size of subsample either as fraction over total samples or as an integer.
        seed (Optional[int], optional): Seed for random number generator. Defaults to None.

    Returns:
        np.ndarray: Indices to be used in subsample
    """
    if seed is None:
        rng = np.random.mtrand._rand
    else:
        rng = np.random.RandomState(seed)

    if max_samples is None:
        max_samples = n_samples
    elif type(max_samples) == float:
        max_samples = round(max_samples * n_samples)

    sample_indices = rng.randint(0, n_samples, max_samples)
    return sample_indices
    



class MonotoneTreeRegressor:
    def __init__(
        self,
        n_trees: int = 100,
        alpha: Union[list, np.ndarray, None] = None,
        fit_intercept: bool = False
        ) -> None:
        """_summary_

        Args:
            n_trees (int, optional): number of trees to use in random forest embedding. Defaults to 100.
            alpha (Union[list, np.ndarray, None], optional): regularization parameter 
                                                            (if list or array then use GCV formula to select best alpha). Defaults to None.
        """
        
        self._n_trees = n_trees 
        self._alpha = alpha 
        self._fit_intercept = fit_intercept
        self._rfe = RandomTreesEmbedding(n_estimators=self._n_trees)
        self._embedding = None
        self._ridge_reg = RidgeCV(alphas=self._alpha, fit_intercept=self._fit_intercept, gcv_mode="svd")
        

    def fit(self, X, y, w = None):
        self._rfe.fit(X)
        self._embedding = self._rfe.transform(X)

        embedding_weighted, y_weighted = _scale_data(self._embedding, y, w=w)

        _ = self._ridge_reg.fit(X = embedding_weighted, y = y_weighted)

    def predict(self, X):
        
        new_embedding = self._rfe.transform(X)

        y_hat = self._ridge_reg.predict(new_embedding)

        return y_hat



class ConstrainedMonotoneTreeRegressor:
  
    def __init__(
        self,
        n_trees: int = 100,
        max_samples: Union[int, float, None] = None,
        n_estimators: int = 1,
        partial_order: Callable = None, 
        cv: Optional[int] = None, # make name more intuitive
        alpha: Union[list, np.ndarray, float, int] = 1,
        use_slack: bool = False,
        gamma: Union[list, np.ndarray, float, int] = None,
        fit_intercept: bool = False,
        n_jobs: int = None,
        seed: Optional[int] = None,
        verbose: int = 0
        ) -> None:
        """_summary_

        Args:
            n_trees (int, optional): number of trees to use in random forest embedding. Defaults to 100.
            max_samples (Union[int, float, None], optional): Number of subsamples or % ot total samples to use 
                                                            when subsampling and aggregating. Defaults to None.
            n_estimators (int, optional): number of estimators to fit for 
                                         subsampling and aggregating. Defaults to 1.
            alpha (Union[list, np.ndarray, None], optional): regularization parameter 
                                                            (if list or array then use GCV formula to select best alpha). Defaults to None.
            n_jobs (int, optional): number of jobs for parallel executio. Defaults to None.
            seed (Optional[int], optional): random seed. Defaults to None.
            verbose (int, optional): verbose parameter for joblib parallel . Defaults to 0.
        """
        
        self._n_trees = n_trees  
        self._max_samples = max_samples
        self._n_estimators = n_estimators

        self._partial_order = partial_order

        self._alpha = alpha 
        self._use_slack = use_slack
        self._gamma = gamma
        self._fit_intercept = fit_intercept
        
        self._cv = cv

        self._n_jobs  = n_jobs 
        self._seed    = seed 
        self._verbose = verbose 

        self._rfe = RandomTreesEmbedding(n_estimators=self._n_trees)

        # Initialize future parameters
        self._embedding = None
        self._w = None # weights for weighted regression
        self._ridge_reg = None
        self._quadratic_term = None
        self._linear_term = None
    
    def fit(self, X, y, w=None):
        self._rfe.fit(X)
       
        if isinstance(type(w), type(None)):
            self._w = np.ones_like(y)


        embedding = self._rfe.transform(X)
        self._embedding, self._y = _scale_data(embedding, y, w=w)

        # Use GCV and ridge without surrogate constraints to retreive optimal alpha
        self._ridge_reg = RidgeCV(alphas=self._alpha, fit_intercept=self._fit_intercept, gcv_mode="svd")
        _ = self._ridge_reg.fit(X=self._embedding, y=self._y)

        # beta_wghts: reqeight the penalization in ridge regression by the corresponding coefficient from OLS
        #             we create a matrix where the ith diagonal element is the inverse of the ith ols coefficient.
        # cross validation 
        self._beta_wghts  = csr_matrix(np.diag(1/self._ridge_reg.coef_**2))
        self._alpha_gcv = self._ridge_reg.alpha_
        self._alpha_space = np.array([2**i for i in [-1,0,1]])*self._alpha_gcv


        n_samples = X.shape[0]

        # retreive subsample indices for subsampling and aggregating
        sample_indices = [_get_sample_indices(n_samples, self._max_samples, self._seed) for i in
                          range(self._n_estimators)]


        # get constraints used in quadratic program
        # self._constr_matrix, self._constr_vec, self._surrogate_constr_idx = get_constraints(embedding=self._embedding, surrogate_outcome=X, partial_order=self._partial_order)
        self._constr_matrix, self._constr_vec, self._surrogate_constr_idx = get_constraints_mp(
            embedding=self._embedding,
            surrogate_outcome=X, 
            partial_order=self._partial_order,
            cpus=self._n_jobs)
        
        if self._constr_matrix is None:
            print("No two surrogate outcomes were comparable, defaulting to unconstrained regression with monotone tree embeddeding.")
            

        else:
            # When constraints exists, fit constrained quadratic program 

            if type(self._alpha) is float:
                if self._cv is not None:
                    print("CV folds was specified but only one alpha value was provided. Omitting cross validation step.\n")

                self._cv = None 

            self._quadratic_term = self._embedding.T @ self._embedding
            self._linear_term = -2 * self._embedding.T @ self._y

            if self._cv is None:
                self._montone_tree_estimators = Parallel(
                    n_jobs=self._n_jobs,
                    verbose=self._verbose,
                    prefer="threads",
                )(delayed(fit_monotone_tree)(
                    surrogate_outcome=X[sample_indices[i]].copy(),
                    embedding=self._embedding[sample_indices[i]].copy(),
                    outcome=y[sample_indices[i]].copy(),
                    partial_order=self._partial_order,
                    alpha=self._alpha,
                    beta_wghts = self._beta_wghts,
                    use_slack = self._use_slack,
                    gamma = self._gamma
                )
                    for i in range(self._n_estimators)
                )
            else:
                kf = KFold(n_splits=self._cv)

                # If only one gamma is provided
                if not hasattr(self._gamma, '__iter__'):
                    self._gamma = [self._gamma]

                params_list = [dict(alpha = params[0], gamma = params[1]) \
                    for params in itertools.product(self._alpha_space, self._gamma)]

                cv_values = np.zeros(shape=(len(params_list), self._cv))
                counter = 0
                for train, test, in kf.split(X):
                    embedding_train, embedding_test, y_train, y_test = self._embedding[train, :], self._embedding[test, :], self._y[train], self._y[test]
                    X_train = X[train,:]

                    # Subset the constraint matrix by observations that occur in training set
                    surrogate_index = np.isin(self._surrogate_constr_idx, train)
                    constr_matrix_cv, constr_vec_cv = self._constr_matrix[surrogate_index,:], self._constr_vec[surrogate_index]

                    if isinstance(self._max_samples, float):
                        n_frac =  self._max_samples 
                    else:
                        n_frac = self._max_samples / n_samples

                    sample_indices_cv = [_get_sample_indices(X_train.shape[0], n_frac, self._seed) for i in
                            range(self._n_estimators)]

                    for k, params in enumerate(params_list):
                        estimators = Parallel(
                            n_jobs=self._n_jobs,
                            verbose=self._verbose,
                            prefer="threads",
                        )(delayed(fit_monotone_tree)(
                            surrogate_outcome=X_train[sample_indices_cv[i]].copy(),
                            embedding=embedding_train[sample_indices_cv[i]].copy(),
                            outcome=y_train[sample_indices_cv[i]].copy(),
                            partial_order=self._partial_order,
                            constr_matrix = constr_matrix_cv,
                            constr_vec = constr_vec_cv,
                            alpha=params["alpha"],
                            beta_wghts = self._beta_wghts,
                            use_slack = self._use_slack,
                            gamma = params["gamma"]
                        )
                            for i in range(self._n_estimators)
                        )

                        y_hat = np.mean([estimator(embedding_test) for estimator in estimators], axis=0)
                    

                        cv_values[k, counter] = mean_squared_error(y_test, y_hat)

                    # increase index for cv iterations
                    counter += 1

                cv_mses = cv_values.mean(axis=1)
                best = np.argmin(cv_mses)
                self._optimal_alpha = params_list[best]["alpha"]
                self._optimal_gamma = params_list[best]["gamma"]
                
                self._montone_tree_estimators = Parallel(
                    n_jobs=self._n_jobs,
                    verbose=self._verbose,
                    prefer="threads",
                )(delayed(fit_monotone_tree)(
                    surrogate_outcome=X[sample_indices[i]].copy(),
                    embedding=self._embedding[sample_indices[i]].copy(),
                    outcome=y[sample_indices[i]].copy(),
                    partial_order=self._partial_order,
                    constr_matrix = self._constr_matrix, 
                    constr_vec = self._constr_vec,
                    alpha=self._optimal_alpha,
                    beta_wghts = self._beta_wghts,
                    use_slack = self._use_slack,
                    gamma = self._optimal_gamma
                )
                    for i in range(self._n_estimators)
                )

    
    def predict(self, X):

        embedding = self._rfe.transform(X)

        if self._constr_matrix is None:
            print("Warning: Using unconstrained regression with monotone tree embedding")
            y_hat = self._ridge_reg.predict(embedding)
        else:
            y_hat = np.mean([estimators(embedding) for estimators in self._montone_tree_estimators], axis=0)

        return y_hat


        
        



if __name__ == "__main__":
    # import pandas as pd
    import numpy as np
    from src.orders import product_order

    from bandit_data import BanditData
    from helper_functions import generate_monotone_linear_spline as gen_gx
    from helper_functions import uniform_exploration as gen_pix

    import numpy.random as nr
    from scipy.stats import norm, expon

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    from policy import *
    from generative_models import *
    import time

    import matplotlib.pyplot as plt
    import xgboost as xgb
    # from xgboost import XGBRegressor

    # Set parameters for sampling model for bandit data
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


    # Setting parameters for estimators

    # Monotone embedding 
    n_trees = 100
    alphas = 10**np.linspace(3,-2,100)

    # Paramaters specifically for constrained monotone tree regressor
    n_estimators = 10
    # alphas = [0.01, 1, 10]
    max_samples = .7
    n_estimators = 5
    cv = 5

   # create instance of monotone tree regressor
    mtr = MonotoneTreeRegressor(
        n_trees = n_trees,
        alpha = alphas
    )

    # create instance of constrained monotone tree regressor
    mtr_constrained = ConstrainedMonotoneTreeRegressor(
        n_trees = n_trees,
        alpha = alphas,
        max_samples = max_samples,
        n_estimators = n_estimators,
        partial_order = product_order,
        cv = cv
    )

    mtr.fit(X = bandit._surrogate, y = bandit._outcome)

    start = time.time()
    mtr_constrained.fit(X = bandit._surrogate, y = bandit._outcome)
    print(f"Elapsed time: {((time.time()-start )/60)}")


    new_sample = gen_model.gen_sample(n)

    mtr.predict(X = new_sample._surrogate)
    mtr_constrained.predict(X = new_sample._surrogate)





    mtr_constrained = ConstrainedMonotoneTreeRegressor(
        n_trees = n_trees,
        alpha = alphas,
        max_samples = max_samples,
        n_estimators = n_estimators,
        partial_order = product_order,
        use_slack=True,
        gamma = [0.01, 1, 10, 100],
        cv = cv
    )


    start = time.time()
    mtr_constrained.fit(X = bandit._surrogate, y = bandit._outcome)
    print(f"Elapsed time: {((time.time()-start )/60)}")
