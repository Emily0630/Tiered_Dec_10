import pdb
import time
from unittest.mock import CallableMixin
from joblib import Parallel, delayed
import numpy.random as npr
import numpy as np
from scipy.stats import expon
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from typing import List, Callable, Union, Dict

# Local Module
from src.bandit_data import BanditData
from src.monotonic_tree import *
from src.helper_functions import *


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


class RandomAction:
    """
    Selections action randomly from available treatments
    """

    def __init__(self, actions):
        self._actions = actions

    def __str__(self):
        return "RandomAction"

    def pick_action(self, n):
        return npr.choice(self._actions, size=n, replace=True)


class MonotoneTreeBootEG:
    """
    Class for bandit estimators using high dimensional feature representation induce by Random Forests and constrained regression
    according to partial order over the observed surrogate outcomes. This stategy uses epsilon greedy.
    """

    def __init__(
            self,
            actions: Union[np.ndarray, List],
            n_trees: int = 100,
            max_samples: Union[int, float, None] = None,
            n_estimators: int = 1,
            alpha: Union[list, np.ndarray, None] = None,
            cv_folds: int = None,
            n_jobs: int = None,
            seed: Optional[int] = None,
            model=None,
            model_params: Dict = None,
            verbose: int = 0
    ) -> None:

        self.actions = np.array(actions)
        self.n_actions = len(actions)
        # self.action_estimates = {action: None for action in actions}
        self._alpha = alpha
        self._cv_folds = cv_folds
        self._n_trees = n_trees
        self._rfe = RandomTreesEmbedding(n_estimators=self._n_trees)
        self._n_estimators = n_estimators
        self._max_samples = max_samples
        self._tree_embedding = None
        self._embedding = None
        self._n_jobs = n_jobs
        self._boot_wghts = None
        self._seed = seed
        self._verbose = verbose

        if self._n_jobs is None:
            self._n_jobs = mp.cpu_count() - 1

        assert 'fit' in dir(model) and 'predict' in dir(model)

        if model_params is not None:
            self.action_estimates = {action: model(*model_params) for action in self.actions}
        else:
            self.action_estimates = {action: model() for action in self.actions}


    def __str__(self):
        return 'MonotoneTreeBootEG'

    def __repr__(self):
        return (f'MonotoneTreeBootEG('
                f'n_trees = {self._n_trees!r}, '
                f'max_samples = {self._max_samples!r}, '
                f'n_estimators = {self._n_estimators!r}, '
                f'alpha = {self._alpha!r}, '
                f'cv_folds = {self._cv_folds!r}, '
                f'n_jobs = {self._n_jobs!r}, '
                f'seed = {self._seed!r}, '
                f'verbose = {self._verbose!r})')

    def fit_tree_embedding(self, bandit: BanditData):

        self._tree_embedding = self._rfe.fit(bandit._surrogate)
        self._embedding = self._tree_embedding.transform(bandit._surrogate)
        self._boot_wghts = np.ones(len(bandit))

    
    def estimate_outcome(self, data: BanditData, partial_order: Callable):

        """
        Denoises outcome variable by fitting monotone tree embedded regression constrained by surrogate outcomes
        """

        n_samples = len(data)
        
        # Collect subsample indices use to train each estimator
        sample_indices = [_get_sample_indices(n_samples, self._max_samples, self._seed) for i in
                          range(self._n_estimators)]

        # Estimating outcome with monotonic tree regression using surrogate outcomes through subsampling and aggregation
        self._montone_tree_estimators = Parallel(
            n_jobs=self._n_jobs,
            verbose=self._verbose,
            prefer="threads",
        )(delayed(fit_monotone_tree)(surrogate_outcome=data._surrogate[sample_indices[i]].copy(),
                                     embedding=self._embedding[sample_indices[i]].copy(),
                                     outcome=data._outcome[sample_indices[i]].copy(),
                                     partial_order=partial_order,
                                     alpha=self._alpha,
                                     wghts=self._boot_wghts[sample_indices[i]].copy()
                                     )
          for i in range(self._n_estimators)
          )



    def estimate_action_value(self, data, y_hat):
        """
        Estimates value of a action by fitting regression on historical data subsetted by that action.
        """
        

        assert len(y_hat) == data._context.shape[0], f"y_hat should have dimesnon {data._context.shape[0]} but has {y_hat.shape[0]}"

        # Estimating action value by fitting regression on subset data where action was given.
        Parallel(
            n_jobs=self._n_jobs,
            verbose=self._verbose,
            prefer="threads",
        )(delayed(
            self.action_estimates[action].fit)(X=data._context[data._action == action].copy(),
                                         y=y_hat[data._action == action].copy()
                                         )
            for action in self.actions
            )

        




    def update(self, data: BanditData, partial_order: Callable):
        """
        Updates model according to new data:
            
            1.) Fit randforest embedding of surrogate outcomes
            2.) Construct esimate of outcome using quadraicaly constrained program with constrains induced by partial order on surrogate outcome
            3.) Build regression model that esimates value of each action
        """

        self.fit_tree_embedding(data)

        assert np.array_equal(np.unique(self.actions), np.unique(data._action))
        # assert self._tree_embedding is not None, "Model has not fit. Please run 'fit_tree_embedding'"

        # Outcome estimation using monotonic tree embedding
        self.estimate_outcome(data, partial_order)

        # Aggregating outcome predictions
        self._y_hat = np.mean([estimator(self._embedding) for estimator in self._montone_tree_estimators], axis=0)

        # Regression model for each treatment
        self.estimate_action_value(data, y_hat = self._y_hat)

        # self.action_estimates = {action: estimators[i] for i, action in enumerate(self.actions)}

   

    def pick_action(self, context: np.ndarray) -> np.ndarray:
        """Maps context to action by selecting action with largest estimated outcome.

        Args:
            context (np.ndarray): n x p matrix represent p dimensional context variables for n indivudals

        Returns:
            np.ndarray: n dimensional vector of actions
        """

        assert self._tree_embedding is not None, "Model has not fit. Please run 'fit_tree_embedding'"

        n = context.shape[0]
        predictions = np.zeros((n, self.n_actions))

        # Estimate outcome under each action
        for i, action in enumerate(self.actions):
            predictions[:, i] = self.action_estimates[action].predict(context)

        # Select action with largest estimated outcome
        optimal_action = self.actions[predictions.argmax(1)]

        return optimal_action




class MonotoneTreeBootTS:
    """
    Class for bandit estimators using high dimensional feature representation induce by Random Forests and constrained regression
    according to partial order over the observed surrogate outcomes. This stategy uses bootstrap thompson sampling.
    """

    def __init__(
            self,
            actions: Union[np.ndarray, List],
            n_trees: int = 100,
            max_samples: Union[int, float, None] = None,
            n_estimators: int = 1,
            alpha: Union[list, np.ndarray, None] = None,
            cv_folds: int = None,
            n_jobs: int = None,
            seed: Optional[int] = None,
            model=None,
            model_params: Dict = None,
            verbose: int = 0
    ) -> None:

        self.actions = np.array(actions)
        self.n_actions = len(actions)
        # self.action_estimates = {action: None for action in actions}
        self._alpha = alpha
        self._cv_folds = cv_folds
        self._n_trees = n_trees
        self._rfe = RandomTreesEmbedding(n_estimators=self._n_trees)
        self._n_estimators = n_estimators
        self._max_samples = max_samples
        self._tree_embedding = None
        self._embedding = None
        self._n_jobs = n_jobs
        self._boot_wghts = None
        self._seed = seed
        self._verbose = verbose

        if self._n_jobs is None:
            self._n_jobs = mp.cpu_count() - 1

        assert 'fit' in dir(model) and 'predict' in dir(model)

        if model_params is not None:
            self.action_estimates = {action: model(*model_params) for action in self.actions}
        else:
            self.action_estimates = {action: model() for action in self.actions}


    def __str__(self):
        return 'MonotoneTreeBootTS'

    def __repr__(self):
        return (f'MonotoneTreeBootTS('
                f'n_trees = {self._n_trees!r}, '
                f'max_samples = {self._max_samples!r}, '
                f'n_estimators = {self._n_estimators!r}, '
                f'alpha = {self._alpha!r}, '
                f'cv_folds = {self._cv_folds!r}, '
                f'n_jobs = {self._n_jobs!r}, '
                f'seed = {self._seed!r}, '
                f'verbose = {self._verbose!r})')

    def fit_tree_embedding(self, bandit: BanditData):

        self._tree_embedding = self._rfe.fit(bandit._surrogate)
        self._embedding = self._tree_embedding.transform(bandit._surrogate)
        self._boot_wghts = expon.rvs(scale=1, size=len(bandit))

    
    def estimate_outcome(self, data: BanditData, partial_order: Callable):

        """
        Denoises outcome variable by fitting monotone tree embedded regression constrained by surrogate outcomes
        """

        n_samples = len(data)
        
        # Collect subsample indices use to train each estimator
        sample_indices = [_get_sample_indices(n_samples, self._max_samples, self._seed) for i in
                          range(self._n_estimators)]

        # Estimating outcome with monotonic tree regression using surrogate outcomes through subsampling and aggregation
        self._montone_tree_estimators = Parallel(
            n_jobs=self._n_jobs,
            verbose=self._verbose,
            prefer="threads",
        )(delayed(fit_monotone_tree)(surrogate_outcome=data._surrogate[sample_indices[i]].copy(),
                                     embedding=self._embedding[sample_indices[i]].copy(),
                                     outcome=data._outcome[sample_indices[i]].copy(),
                                     partial_order=partial_order,
                                     alpha=self._alpha,
                                     wghts=self._boot_wghts[sample_indices[i]].copy()
                                     )
          for i in range(self._n_estimators)
          )



    def estimate_action_value(self, data, y_hat):
        """
        Estimates value of a action by fitting regression on historical data subsetted by that action.
        """
        

        assert len(y_hat) == data._context.shape[0], f"y_hat should have dimesnon {data._context.shape[0]} but has {y_hat.shape[0]}"

        # Estimating action value by fitting regression on subset data where action was given.
        Parallel(
            n_jobs=self._n_jobs,
            verbose=self._verbose,
            prefer="threads",
        )(delayed(
            self.action_estimates[action].fit)(X=data._context[data._action == action].copy(),
                                         y=y_hat[data._action == action].copy()
                                         )
            for action in self.actions
            )

        




    def update(self, data: BanditData, partial_order: Callable):
        """
        Updates model according to new data:
            
            1.) Fit randforest embedding of surrogate outcomes
            2.) Construct esimate of outcome using quadraicaly constrained program with constrains induced by partial order on surrogate outcome
            3.) Build regression model that esimates value of each action
        """

        self.fit_tree_embedding(data)

        assert np.array_equal(np.unique(self.actions), np.unique(data._action))
        # assert self._tree_embedding is not None, "Model has not fit. Please run 'fit_tree_embedding'"

        # Outcome estimation using monotonic tree embedding
        self.estimate_outcome(data, partial_order)

        # Aggregating outcome predictions
        self._y_hat = np.mean([estimator(self._embedding) for estimator in self._montone_tree_estimators], axis=0)

        # Regression model for each treatment
        self.estimate_action_value(data, y_hat = self._y_hat)

        # self.action_estimates = {action: estimators[i] for i, action in enumerate(self.actions)}

   

    def pick_action(self, context: np.ndarray) -> np.ndarray:
        """Maps context to action by selecting action with largest estimated outcome.

        Args:
            context (np.ndarray): n x p matrix represent p dimensional context variables for n indivudals

        Returns:
            np.ndarray: n dimensional vector of actions
        """

        assert self._tree_embedding is not None, "Model has not fit. Please run 'fit_tree_embedding'"

        n = context.shape[0]
        predictions = np.zeros((n, self.n_actions))

        # Estimate outcome under each action
        for i, action in enumerate(self.actions):
            predictions[:, i] = self.action_estimates[action].predict(context)

        # Select action with largest estimated outcome
        optimal_action = self.actions[predictions.argmax(1)]

        return optimal_action


def _policy_value(policy_measure, Y: np.ndarray) -> float:
    """Returns value of policy

    Args:
        policy_measure (_type_): measure iduced by that policy
        Y (np.ndarray): observed outcomes

    Returns:
        float: value of policy
    """
    assert len(policy_measure) == len(Y), "Measure and outcome must be same dimension"

    val = np.sum(policy_measure * Y)

    return val


class IPW:
    """
    Bandit algorithm that uses Inverse proability Weighting (IPW) estimator for policy value. 
    Optimal policy is policy with the largest estimated value.
    """
    def __init__(self,
                 policies: List,
                 n_jobs: int = None,
                 verbose: int = 0
                 ) -> None:
        self._policies = policies
        self._n_jobs = n_jobs
        self._verbose = verbose

        if self._n_jobs is None:
            self._n_jobs = mp.cpu_count() - 1

        return None

    def get_opt(self, data: BanditData, w: Optional[np.ndarray] = None) -> Callable:
        """
        Returns policy that optimizes expected value of Y
        :param data: bandit data
        :param w: vector of weights used in bootstrap
            thompson sampling for generating policy measure
        :return: optimal policy
        """

        if w is None:
            w = np.ones(len(data))

        assert len(data) == len(w), "Expected w array of dim ({0},) but received ({1},)".format(len(data), len(w))

        policy_values = Parallel(
            n_jobs=self._n_jobs,
            verbose=self._verbose,
            prefer="threads",
        )(delayed(_policy_value)(policy_measure=_policy_to_measure(self._policies[i], data, w=w),
                                 Y=data._outcome
                                 )
          for i in range(len(self._policies))
          )

        optimal_policy = self._policies[np.argmax(policy_values)]

        return optimal_policy

    def update(self, data: BanditData):

        self.optimal_policy = self.get_opt(data=data)


class IpwEpsGreedy(IPW):
    """
    Uses epsilon greedy algorithm for action selection.

    Args:
        IPW (_type_): Inherits from IPW class
    """

    def __init__(self,
                 policies: List,
                 n_jobs: int = None,
                 verbose: int = 0
                 ) -> None:
        super().__init__(policies, n_jobs, verbose)

    def __str__(self):
        return 'IpwEpsGreedy'

    def __repr__(self):
        return f'IpwEpsGreedy(n_jobs = {self._n_jobs!r})'

    def random_policy(self):
        pol_id = int(npr.choice(len(self._policies), 1))
        return self._policies[pol_id]

    def pick_action(
            self,
            context: np.ndarray,
            epsilon: Union[float, np.ndarray] = 0.0
    ) -> np.ndarray:
        """
        Implements epsilon greedy for action selection: with probability 1-epsilon assign action
        according to optimal policy, otherwise assign action uniformly.

        Args:
            context (np.ndarray): context features 
            epsilon (Union[float, np.ndarray], optional): Probability that determines whether the optimal policy 
            is used or an action is assigned uniformly. If a scalar is assigned it represents one probability that is 
            used for each observation. Otherwise it expects a vector probabilities that corresponds to the 
            dimension of X. Defaults to 0.0.

        Returns:
            _type_: _description_
        """

        if isinstance(epsilon, float):
            epsilon = np.ones(len(context)) * epsilon
        else:
            assert len(context) == len(epsilon), "Expected either single epsilon or an array of the same length as X"

        new_actions = [] 
        propensity = []
        for i, x in enumerate(context):
            choose_random = npr.random() < epsilon[i]
            if choose_random:
                policy = self.random_policy()
                new_a = policy.decision(x)
                propensity.append(epsilon[i]) # probability action was assigned

            else:
                new_a = self.optimal_policy.decision(x)
                propensity.append(1 - epsilon[i]) # probability action was assigned

            new_actions.append(new_a)

        return np.array(new_actions), np.array(propensity)


class IpwBootTS(IPW):
    """
    Uses bootstrap thompson sampling with IPW estiamtor for estimation of policy value and optimal policy.
    Args:
        IPW (_type_): Inherits from IPW
    """
    def __init__(self,
                 policies: List,
                 n_jobs: int = None,
                 verbose: int = 0,
                 replicates: int = 10000
                 ) -> None:
        super().__init__(policies, n_jobs, verbose)

        self._num_policies = len(policies)
        self._replicates = replicates
        self._replicate_opt_policy_id = np.zeros(self._replicates).astype(int)

    def __str__(self):
        return 'IpwBootTS'

    def __repr__(self):
        return f'IpwBootTS(n_jobs = {self._n_jobs!r}, replicates = {self._replicates!r})'

    def update(self, data: BanditData):
        """For J replicates estimates the policy value using a boostrapped approximation
        using exponential(1) weights. For each replicate assign the optimal policy.

        Args:
            data (BanditData): Bandit data
        """

        for J in range(self._replicates):
            outcomes = np.zeros(self._num_policies)
            w = expon.rvs(scale=1, size=len(data))
            for i in range(self._num_policies):
                outcomes[i] = _policy_value(
                    _policy_to_measure(self._policies[i], data=data, w=w),
                    data._outcome
                )
            self._replicate_opt_policy_id[J] = int(np.argmax(outcomes))

    def pick_action(self, context: np.ndarray) -> np.ndarray:
        """
        Uses bootstrap thompson sampling to assign action.
        Selects replicate uniformly and assigns optimal policy estimated by that replicate. 
        The estimated propensity score is the fraction of replicates for which the selected policy was 
        estimated to be the optimal policy. 

        Args:
            context (np.ndarray): Context features.

        Returns:
            np.ndarray: Vector of assigned actions.
        """

        new_actions = []
        propensity = []
        for i, x in enumerate(context):
            chosen_replicate = npr.choice(range(self._replicates), 1)[0]
            policy = self._policies[self._replicate_opt_policy_id[chosen_replicate]]
            new_a = policy.decision(x)

            # Probability that policy was chosen
            replicate_a = np.array([
                self._policies[self._replicate_opt_policy_id[i]].decision(x)
                for i in range(self._replicates)
            ])
            new_p = np.mean(new_a == replicate_a)

            propensity.append(new_p)
            new_actions.append(new_a)
        return np.array(new_actions), np.array(propensity)


def _policy_to_measure(policy, data: BanditData, w: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Measure induced by a given policy.

    Args:
        policy (_type_): Map from context to action space.
        data (BanditData): Bandit data
        w (Optional[np.ndarray], optional): Weight vector (used in bootstrap thompson sampling). Defaults to None.

    Returns:
        np.ndarray: measure
    """
    if w is None:
        w = np.ones(len(data))

    assert len(data) == len(w), (
        "Expected w array of dim ({0},) but received ({1},)".format(len(data), len(w)))
    measure = np.zeros(len(data))
    for i, (context, action, surrogate, outcome, propensity) in enumerate(data):
        if action == policy.decision(context):
            measure[i] = (1 / propensity) * w[i]

    measure /= np.sum(measure)
    return measure


class PolicyScreening(IPW):
    """
    Method for estimating an optimal policy when there exits surrogate outcomes that may drive the primary outcome.
    We assume a partial ordering on the surrogate space. Using these assumptins we screen out policies that are dominated 
    by another and estimate the optimal policy using a set of non-dominated policies (consistent with partial order on surrogate outcome).

    Args:
        IPW (_type_): Inherits from IPW.
    """
    check_dominated = {"gurobi": gurobi_check_is_dominated,
                       "cvxopt": cvxopt_check_is_dominated}

    def __init__(
            self,
            policies: List,
            n_jobs: int = -1,
            verbose: int = 0,
            solver: str = 'gurobi'
    ) -> None:

        super().__init__(policies, n_jobs, verbose)

        self._constraint_matrix = None
        self._constraint_vector = None
        self._smoothness_vector = None
        self._smoothness_matrix = None
        check_dominated = {"gurobi": gurobi_check_is_dominated,
                       "cvxopt": cvxopt_check_is_dominated}

        self._non_dominated_indices = None
        assert solver in check_dominated.keys()
        self._solver = solver

        return None

    def build_constraints(
            self,
            data: BanditData,
            partial_order: Callable
    ) -> None:
        """
        Constructs sparse constraint matrix used for linear program that evaluates if one policy dominates another 

        Args:
            data (BanditData): bandit data
            partial_order (Callable): a binary function that defines a partial order and evaluates whether one object is greater than another.
        """

        n = len(data._action)
        self._constraint_matrix = dok_matrix((
            int(n_choose_k(n, 2)) + 2 * n,
            n
        ))

        self._smoothness_matrix = dok_matrix((
            int(2*n_choose_k(n, 2)),
            n
        ))

        self._constraint_vector = np.zeros(int(n_choose_k(n, 2)) + 2 * n)
        self._smoothness_vector = np.zeros(2*int(n_choose_k(n, 2)))
        counter = 0
        smoothness_counter = 0
        p_sur = data._surrogate.shape[1]
        for i in range(n):
            self._constraint_matrix[counter, i] = 1
            self._constraint_vector[counter] = 1
            self._constraint_matrix[counter + 1, i] = -1
            counter += 2
            for j in range(i):
                ## Smoothness
                dij = np.linalg.norm(
                    data._surrogate[i, :] - data._surrogate[j, :]
                )

                self._smoothness_matrix[smoothness_counter, i] = 1
                self._smoothness_matrix[smoothness_counter, j] = -1
                self._smoothness_vector[smoothness_counter] = np.sqrt(p_sur)*dij
                smoothness_counter += 1

                self._smoothness_matrix[smoothness_counter, i] = -1
                self._smoothness_matrix[smoothness_counter, j] = 1
                self._smoothness_vector[smoothness_counter] = np.sqrt(p_sur)*dij
                smoothness_counter += 1

                compare = partial_order(
                    data._surrogate[i, :],
                    data._surrogate[j, :]
                )
                # compare = 
                if compare == 0:
                    continue
                self._constraint_matrix[counter, i] = compare
                self._constraint_matrix[counter, j] = -compare
                counter += 1



        self._constraint_matrix = self._constraint_matrix[0:counter, ]
        self._constraint_vector = self._constraint_vector[0:counter]

    # compare policies to get non dominated policies
    def search_non_dominated(
            self,
            data: BanditData,
            w: Optional[np.ndarray] = None
    ) -> None:
        """
        Finds set of non dominated policies according to partial order over surrogate outcomes.

        Args:
            data (BanditData): bandit data
            w (Optional[np.ndarray], optional): Weight vector used for bootstrap thompson sampling. Defaults to None.
        """
        #### TESTING
        #start = time.time()
        #n = len(self._policies)
        #op = Parallel(n_jobs=-1)(delayed(check_i_is_dominated)(i, self, data, w) for i in range(n))
        #print("Parallel took: ", time.time() - start)
        #### END TESTING

        start = time.time()
        assert self._constraint_vector is not None, "Must build constraints first"

        if w is None:
            w = np.ones(len(data))

        assert len(data) == len(w), "Expected w array of dim ({0},) but received ({1},)".format(len(data), len(w))

        non_dominated_indices = set(range(len(self._policies)))
        #pdb.set_trace()
        for i in range(len(self._policies)):
            i_is_dominated = False
            for j in non_dominated_indices:
                if i == j:
                    continue
                delta = _policy_to_measure(self._policies[j], data, w=w) - \
                        _policy_to_measure(self._policies[i], data, w=w)

                i_is_dominated = self.check_dominated[self._solver](
                    delta,
                    self._constraint_matrix,
                    self._constraint_vector,
                    self._smoothness_matrix,
                    self._smoothness_vector
                )
                if i_is_dominated:
                    non_dominated_indices.remove(i)
                    break
        print("serial took: ", time.time() - start)
        print("num-non-dominated: " + str(len(non_dominated_indices)))
        #pdb.set_trace()
        self._non_dominated_indices = list(non_dominated_indices)

    def search_non_dominated_no_mutate(
            self,
            data: BanditData,
            w: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Finds set of non dominated policies according to partial order over surrogate outcomes.

        Args:
            data (BanditData): bandit data
            w (Optional[np.ndarray], optional): Weight vector used for bootstrap thompson sampling. Defaults to None.
        """

        assert self._constraint_vector is not None, "Must build constraints first"

        if w is None:
            w = np.ones(len(data))

        assert len(data) == len(w), "Expected w array of dim ({0},) but received ({1},)".format(len(data), len(w))

        non_dominated_indices = set(range(len(self._policies)))

        for i in range(len(self._policies)):
            i_is_dominated = False
            for j in non_dominated_indices:
                if i == j:
                    continue
                delta = _policy_to_measure(self._policies[j], data, w=w) - \
                        _policy_to_measure(self._policies[i], data, w=w)

                i_is_dominated = self.check_dominated[self._solver](
                    delta,
                    self._constraint_matrix,
                    self._constraint_vector
                )
                if i_is_dominated:
                    non_dominated_indices.remove(i)
                    break
        print("num-non-dominated: " + str(len(non_dominated_indices)))
        return list(non_dominated_indices)


    # choose optimal policy from dominated set
    def get_non_dominated_opt(self, data: BanditData, w: Optional[np.ndarray] = None):
        """
        Estimates optimal policy from set of non dominated policies.

        Args:
            data (BanditData): Bandit data
            w (Optional[np.ndarray], optional):  Weight vector used for bootstrap thompson sampling. Defaults to None.

        Returns:
            _type_: optimal policy
        """

        if w is None:
            w = np.ones(len(data))

        assert len(data) == len(w), (
            "Expected w array of dim ({0},) but received ({1},)".format(
                len(data), len(w)
            )
        )


        policy_values = Parallel(
            n_jobs=self._n_jobs,
            verbose=self._verbose,
            prefer="threads",
        )(delayed(_policy_value)(
            _policy_to_measure(
                policy=self._policies[i],
                data=data,
                w=w
            ),
            Y=data._outcome
        )
          for i in self._non_dominated_indices
        )

        optimal_policy = self._policies[
            self._non_dominated_indices[np.argmax(policy_values)]
        ]

        return optimal_policy

    def update(self, data: BanditData, partial_order: Callable):
        """
        Generates constrains, set of non dominated policies, and optimal policy from newly observed data.

        Args:
            data (BanditData): bandit data.
            partial_order (Callable): a binary function that defines a partial order and evaluates whether one object is greater than another.
        """

        self.build_constraints(data, partial_order)
        self.search_non_dominated(data)
        self.optimal_policy = self.get_non_dominated_opt(data)


class PolicyScreeningEpsGreedy(PolicyScreening):
    """
    Epsilon greedy algorithm used with policy screening.

    Args:
        PolicyScreening (_type_): Inherits from PolicyScreening.
    """

    def __init__(
            self,
            policies: List,
            n_jobs:  int = None,
            verbose: int = 0,
            solver:  str = 'gurobi'
    ) -> None:
        super().__init__(policies, n_jobs, verbose, solver)

    def __str__(self):
        return 'PolicyScreeningEpsGreedy'

    def __repr__(self):
        return f'PolicyScreeningEpsGreedy(n_jobs = {self._n_jobs!r})'

    def random_policy(self):
        pol_id = npr.choice(len(self._policies), 1)[0]
        return self._policies[pol_id]

    def pick_action(
            self,
            X: np.ndarray,
            epsilon: Union[float, np.ndarray] = 0.0,
            n_action: float=2.0
    ) -> np.ndarray:
        """
        Assigns action according to epsilon greedy with policy screening algorithm.
        At any given decision assigns an action according to the optimal policy
        estimated from set of non dominated policies with probabiliy 1 - epsilon,
        otherwise it assigns an action from a policy selected uniformly from
        the class of policies.

        Args:
            X (np.ndarray): bandit data.
            epsilon (Union[float, np.ndarray], optional): Probability that
            determines whether the optimal policy is used or an action is
            assigned uniformly. If a scalar is assigned it represents one
            probability that is used for each observation. Otherwise it
            expects a vector probabilities that corresponds to the
            dimension of X. Defaults to 0.0.
            n_action float: number of actions

        Returns:
            np.ndarray: Vector of actions
        """

        if isinstance(epsilon, float):
            epsilon = np.ones(len(X)) * epsilon
        else:
            assert len(X) == len(epsilon),\
                "Expected either single epsilon or an array of the same length as X"

        new_actions = []
        new_propensities = []
        for i, x in enumerate(X):
            choose_random = npr.random() < epsilon[i]
            if choose_random:
                policy = self.random_policy()
                new_a = policy.decision(x)
                new_prop = epsilon/n_action

            else:
                new_a = self.optimal_policy.decision(x)
                new_prop = 1-epsilon + epsilon/n_action

            new_actions.append(new_a)
            new_propensities.append(new_prop)

        return np.array(new_actions), np.array(new_propensities)



class PolicyScreeningBootTS(PolicyScreening):
    """
    Bootstrap thompson sampling with policy screen algorithm.

    Args:
        PolicyScreening (_type_): Inherits from PolicyScreening.
    """
    def __init__(
            self,
            policies: List,
            n_jobs: int = None,
            verbose: int = 0,
            replicates: int = 100,
            solver: str = 'gurobi'
    ) -> None:
        super().__init__(policies, n_jobs, verbose, solver)

        self._optimal_policy_non_dominated = None
        self._optimal_policy = None
        self._w = None
        self._num_policies = len(policies)
        self._replicates = replicates
        self._replicate_opt_policy_id = np.zeros(self._replicates).astype(int)
        self._replicate_opt_policies = []

    def __str__(self):
        return 'PolicyScreeningBootTS'

    def __repr__(self):
        return f'PolicyScreeningBootTS(n_jobs = {self._n_jobs!r})'

    def update(self, data: BanditData, partial_order: Callable):

        self.build_constraints(data, partial_order)
        self._w = expon.rvs(scale=1, size=len(data))
        self.search_non_dominated(data, w=self._w)
        self.optimal_policy_non_dominated = self.get_non_dominated_opt(data, w=self._w)
        self.optimal_policy = self.get_opt(data, w=self._w)

        #w_j_list = [
        #    expon.rvs(scale=1, size=len(data))
        #    for _ in range(len(data))
        #]
        #self._replicate_opt_policies = Parallel(
        #    n_jobs=self._n_jobs,
        #    verbose=self._verbose,
        #    prefer="threads",
        #)(delayed())

        self._replicate_opt_policies = []
        for j in range(self._replicates):
            w_j = expon.rvs(scale=1, size=len(data))
            self.search_non_dominated(data, w=w_j)
            self._replicate_opt_policies.append(
                self.get_non_dominated_opt(data, w=w_j)
            )
            #print("Finished iteration in prop est: " + str(j))



    def pick_action(
            self,
            X: np.ndarray,
            epsilon: Union[float, np.ndarray] = 0.0
    ):
        """
        Assigns action according to bootstrap thompson sampling with policy screening algorithm.
        At any given decision assigns an action according to the optimal policy estiamted 
        from set of non dominated policies with probabiliy 1 - epsilon, otherwise it assigns 
        an action from the optimal policy estimated from entire policy class. Here a policy value 
        is estimated uses the bootstrapped approximation of the policy value.

        Args:
            X (np.ndarray): bandit data.
            epsilon (Union[float, np.ndarray], optional): Probability that determines whether the optimal policy 
            is used or an action is assigned uniformly. If a scalar is assigned it represents one probability that is 
            used for each observation. Otherwise it expects a vector probabilities that corresponds to the 
            dimension of X. Defaults to 0.0.

        Returns:
            np.ndarray: Vector of actions
        """

        if isinstance(epsilon, float):
            epsilon = np.ones(len(X)) * epsilon
        else:
            assert len(X) == len(epsilon), "Expected either single epsilon or an array of the same length as X"

        new_actions = []
        new_propensities = []
        for i, x in enumerate(X):
            choose_random = npr.random() < epsilon[i]
            if choose_random and False: # no additional explore
                new_a = self.optimal_policy.decision(x)

            else:
                new_a = self.optimal_policy_non_dominated.decision(x)

            new_prop = np.mean([
                policy.decision(x) == new_a
                for policy in self._replicate_opt_policies
            ])

            new_actions.append(new_a)
            new_propensities.append(new_prop)

        print("inside pick action screened TS")
        return np.array(new_actions), np.array(new_propensities)


## Expiremental parallelizable version of policy screening
def check_i_is_dominated(
        i: int,
        policy: PolicyScreening,
        data: BanditData,
        w: Optional[np.ndarray] = None
) -> bool:
    n = len(policy._policies)
    if w is None:
        w = np.ones(len(data))
    for j in range(n):
        if i == j:
            continue
        delta = _policy_to_measure(policy._policies[j], data, w=w) - \
                _policy_to_measure(policy._policies[i], data, w=w)

        i_is_dominated = policy.check_dominated[policy._solver](
            delta,
            policy._constraint_matrix,
            policy._constraint_vector,
            policy._smoothness_matrix,
            policy._smoothness_vector
        )
        if i_is_dominated:
            return True
    return False






if __name__ == "__main__":
    from src.helper_functions import uniform_exploration as gen_pix
    from src.helper_functions import generate_monotone_linear_spline as gen_gx
    from src.generative_models import CopulaGenerativeModel
    from src.policy import LinearBasket
    from src.orders import product_order

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
        g=g,
        log_normal_mean=log_normal_mean,
        log_normal_var=log_normal_var,
        ar_rho=ar_rho,
        x_dim=p,
        num_actions=num_actions,
        action_selection=pi,
        coefficients=theta,
        normal_var=normal_var,
        z_var=z_var,
        mc_iterations=10000
    )

    ## Burn-in
    init_sample = gen_model.gen_sample(n)

    n_trees = 100
    n_estimators = 5

    alphas = 2
    cv_folds = None
    actions = np.arange(num_actions)

    mtbs = MonotoneTreeBootTS(
        actions=actions,
        n_trees=n_trees,
        n_estimators=n_estimators,
        max_samples=.7,
        alpha=.2,
        model = LinearRegression
        
    )

    
    mtbs.update(init_sample, partial_order=product_order)
    x = gen_model.get_context(num_samples=n)
    a = mtbs.pick_action(X=x)

    ## Generate random linear policies
    num_policies = 100
    basket = LinearBasket.generate_random_basket(
        num_policies=num_policies,
        num_actions=num_actions,
        x_dim=p
    )

    # Test regular IPW w/ eps greedy
    eps_greedy = IpwEpsGreedy(policies=basket)
    eps_greedy.update(data=init_sample)

    
    actions_eps, propensities_eps = eps_greedy.pick_action(x, epsilon=.5)

    # Test regular IPW w/ boot ts
    boot_ts = IpwBootTS(policies=basket, replicates=10)
    boot_ts.update(data=init_sample)

    actions_ts, propensities_ts = boot_ts.pick_action(x)

    # Testing epsilon screening
    eps_screening = PolicyScreeningEpsGreedy(policies=basket)

    eps_screening.update(init_sample, partial_order=product_order)

    actions_ep_screen = eps_screening.pick_action(x, epsilon=.5)

    # Testing thompson sampling screening
    ts_screening = PolicyScreeningBootTS(policies=basket)

    ts_screening.update(init_sample, partial_order=product_order)

    actions = ts_screening.pick_action(x, epsilon=.5)
