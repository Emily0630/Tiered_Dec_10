##
## @file generative models.py
## @brief Interface and implementations for generative model object
## @note Joint work with Marc Brooks
## @author Eric B. Laber
##
from __future__ import annotations
import pdb
import numpy as np

import numpy.random as nr
from scipy.stats import norm
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

# Local packages
from src.policy import *
from src.bandit_data import BanditData
from src.helper_functions import multivariate_ar_normal as rmvnorm
from src.helper_functions import generate_monotone_linear_spline as gen_gx
from src.helper_functions import uniform_exploration as gen_pix
from src.helper_functions import cvxopt_check_is_dominated
from src.helper_functions import gurobi_check_is_dominated


class ContextualBandit(ABC):
    @abstractmethod
    def gen_sample(self, num_samples: int) -> BanditData:
        pass


class CopulaGenerativeModel(ContextualBandit):
    def __init__(
            self,
            monotone_func: Callable[[float], float],
            log_normal_mean: float,
            log_normal_var: float,
            normal_var: float,
            autoregressive_cov: float,
            context_ndim: int,
            num_actions: int,
            action_selection: Callable[[np.ndarray], int],
            coefficients: List[np.ndarray],  # one for each action
            surrogate_var: float, # variance for distribution over surrogates
            mc_iterations: int = None
    ) -> None:
        self.optimal_policy = None
        self._monotone_func = monotone_func
        self._log_normal_mean = log_normal_mean
        self._log_normal_var = log_normal_var
        self._log_normal_sd = np.sqrt(log_normal_var)
        self._normal_var = normal_var
        self._normal_sd = np.sqrt(normal_var)
        self._autoregressive_cov = autoregressive_cov
        self._context_ndim = context_ndim
        self._num_actions = num_actions
        self._pi = action_selection
        self._theta = coefficients
        self._surrogate_var = surrogate_var
        self._surrogate_ndim = self._theta[0].shape[1]
        self._mc_iterations = mc_iterations

        assert len(self._theta) == self._num_actions, \
            "dim theta  != num_actions"


    def get_optimal_action(self):
        # Perform MC to estimate optimal action

        assert self._mc_iterations is not None

        print(f"Finding optimal policy using {self._mc_iterations} monte carlo iterations\n")
        action_values = {action: 0 for action in range(self._num_actions)}
        for i in range(self._mc_iterations):
            x = rmvnorm(1, rho=self._autoregressive_cov, dim=self._context_ndim)
            for a in range(self._num_actions):
                z = self.get_surrogates(x, [a])
                y = self.get_outcome(z)
                action_values[a] += float(y)

        self.true_expected_reward = {action: action_values[action] / self._mc_iterations
                                     for action in action_values.keys()
                                     }

        self.optimal_action = max(self.true_expected_reward, key=self.true_expected_reward.get)

    def gen_sample(self, num_samples: int) -> BanditData:
        # Implemented inefficiently for clarity/debugging
        context = rmvnorm(num_samples, rho=self._autoregressive_cov, dim=self._context_ndim)
        action = np.zeros(num_samples, dtype=int)
        propensity = np.zeros(num_samples)
        for i in range(num_samples):
            action[i], propensity[i] = self._pi(context[i, :])
        surrogate = np.zeros((num_samples, self._surrogate_ndim))
        for i in range(num_samples):
            surrogate[i, :] = np.dot(context[i, :], self._theta[action[i]]) + \
                      np.sqrt(self._surrogate_var) * nr.randn(self._surrogate_ndim)

        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            for j in range(self._surrogate_ndim):
                indicator = float(surrogate[i, j] < 0)
                if j > 1:
                    indicator *= np.prod(
                        surrogate[i, 0:j] >= 0
                    )
                outcome[i] += indicator * (self._monotone_func(norm.cdf(surrogate[i, j])) + (self._surrogate_ndim - j))
            outcome[i] = outcome[i] * np.exp(
                nr.randn(1) * self._log_normal_sd + self._log_normal_mean
            ) + self._normal_sd * nr.randn(1)
        return BanditData(context=context, action=action, surrogate=surrogate, outcome=outcome, propensity=propensity)

    def get_context(self, num_samples: int):
        return rmvnorm(num_samples, rho=self._autoregressive_cov, dim=self._context_ndim)

    def get_surrogates(self, context: np.ndarray, action: Union[np.ndarray, list]) -> np.ndarray:

        assert context.shape[0] == len(action), "number of treatments differs from number of samples"

        num_samples = context.shape[0]

        surrogate = np.zeros((num_samples, self._surrogate_ndim))

        for i in range(num_samples):
            surrogate[i, :] = np.dot(context[i, :], self._theta[int(action[i])]) + \
                      np.sqrt(self._surrogate_var) * nr.randn(self._surrogate_ndim)

        return surrogate

    def get_outcome(self, surrogate: np.ndarray):

        num_samples = surrogate.shape[0]

        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            for j in range(self._surrogate_ndim):
                indicator = float(surrogate[i, j] < 0)
                if j > 1:
                    indicator *= np.prod(
                        surrogate[i, 0:j] >= 0
                    )
                outcome[i] += indicator * (self._monotone_func(norm.cdf(surrogate[i, j])) + (self._surrogate_ndim - j))
            outcome[i] = outcome[i] * np.exp(
                nr.randn(1) * self._log_normal_sd + self._log_normal_mean
            ) + self._normal_sd * nr.randn(1)

        return outcome

class MonotoneGenerativeModel(ContextualBandit):
    def __init__(
            self,
            log_normal_mean: float,
            log_normal_var: float,
            normal_var: float,
            autoregressive_cov: float,
            context_ndim: int,
            num_actions: int,
            action_selection: Callable[[np.ndarray], int],
            coefficients_c: List[np.ndarray],  # one for each action,
            coefficients_s: np.ndarray,  # one for each action,
            surrogate_var: float, # variance for distribution over surrogates
            mc_iterations: int = None
    ) -> None:
        self.optimal_policy = None
        self._log_normal_mean = log_normal_mean
        self._log_normal_var = log_normal_var
        self._log_normal_sd = np.sqrt(log_normal_var)
        self._normal_var = normal_var
        self._normal_sd = np.sqrt(normal_var)
        self._autoregressive_cov = autoregressive_cov
        self._context_ndim = context_ndim
        self._num_actions = num_actions
        self._pi = action_selection
        self._coefficients_c = coefficients_c
        self._coefficients_s = coefficients_s

        self._surrogate_var = surrogate_var
        self._surrogate_ndim = self._coefficients_c[0].shape[1]
        self._mc_iterations = mc_iterations

        assert len(self._coefficients_c) == self._num_actions, "dim coefficients_c  != num_actions"
        assert len(self._coefficients_s) == self._surrogate_ndim, "dim of coefficients_s  is inconsistent with dim of coefficients_c"



    def get_optimal_action(self):
        # Perform MC to estimate optimal action

        assert self._mc_iterations is not None

        print(f"Finding optimal policy using {self._mc_iterations} monte carlo iterations\n")
        action_values = {action: 0 for action in range(self._num_actions)}
        for i in range(self._mc_iterations):
            x = rmvnorm(1, rho=self._autoregressive_cov, dim=self._context_ndim)
            for a in range(self._num_actions):
                z = self.get_surrogates(x, [a])
                y = self.get_outcome(z)
                action_values[a] += float(y)

        self.true_expected_reward = {action: action_values[action] / self._mc_iterations
                                     for action in action_values.keys()
                                     }

        self.optimal_action = max(
            self.true_expected_reward,
            key=self.true_expected_reward.get
        )


    def gen_sample(self, num_samples: int) -> BanditData:
        # Implemented inefficiently for clarity/debugging
        context = rmvnorm(num_samples, rho=self._autoregressive_cov, dim=self._context_ndim)
        action = np.zeros(num_samples, dtype=int)
        propensity = np.zeros(num_samples)
        for i in range(num_samples):
            action[i], propensity[i] = self._pi(context[i, :])
        surrogate = np.zeros((num_samples, self._surrogate_ndim))
        for i in range(num_samples):
            surrogate[i, :] = np.dot(context[i, :], self._coefficients_c[action[i]]) + \
                      np.sqrt(self._surrogate_var) * nr.randn(self._surrogate_ndim)

        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            outcome[i] = np.dot(surrogate[i,:], self._coefficients_s) 
            outcome[i] = outcome[i] * np.exp(nr.randn(1) * self._log_normal_sd + self._log_normal_mean) + self._normal_sd * nr.randn(1)
        return BanditData(context=context, action=action, surrogate=surrogate, outcome=outcome, propensity=propensity)

    def get_context(self, num_samples: int):
        return rmvnorm(num_samples, rho=self._autoregressive_cov, dim=self._context_ndim)

    def get_surrogates(self, context: np.ndarray, action: Union[np.ndarray, list]) -> np.ndarray:

        assert context.shape[0] == len(action), "number of treatments differs from number of samples"

        num_samples = context.shape[0]

        surrogate = np.zeros((num_samples, self._surrogate_ndim))

        for i in range(num_samples):
            surrogate[i, :] = np.dot(context[i, :], self._coefficients_c[int(action[i])]) + \
                      np.sqrt(self._surrogate_var) * nr.randn(self._surrogate_ndim)

        return surrogate

    def get_outcome(self, surrogate: np.ndarray):

        num_samples = surrogate.shape[0]

        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            outcome[i] = np.dot(surrogate[i,:], self._coefficients_s) 
            outcome[i] = outcome[i] * np.exp(nr.randn(1) * self._log_normal_sd + self._log_normal_mean) + self._normal_sd * nr.randn(1)
        return outcome


class TestGenerativeModel(ContextualBandit):
    def __init__(
            self,
            monotone_func: Callable[[float], float],
            log_normal_mean: float,
            log_normal_var: float,
            normal_var: float,
            autoregressive_cov: float,
            context_ndim: int,
            num_actions: int,
            action_selection: Callable[[np.ndarray], int],
            coefficients: List[np.ndarray],  # one for each action
            surrogate_var: float, # variance for distribution over surrogates
            mc_iterations: int = None
    ) -> None:
        self.optimal_policy = None
        self._monotone_func = monotone_func
        self._log_normal_mean = log_normal_mean
        self._log_normal_var = log_normal_var
        self._log_normal_sd = np.sqrt(log_normal_var)
        self._normal_var = normal_var
        self._normal_sd = np.sqrt(normal_var)
        self._autoregressive_cov = autoregressive_cov
        self._context_ndim = context_ndim
        self._num_actions = num_actions
        self._pi = action_selection
        self._theta = coefficients
        self._surrogate_var = surrogate_var
        self._surrogate_ndim = self._theta[0].shape[1]
        self._mc_iterations = mc_iterations

        assert len(self._theta) == self._num_actions, \
            "dim theta  != num_actions"


    # This function computes best one-size fits all action
    def get_optimal_action(self):
        # Perform MC to estimate optimal action
        assert self._mc_iterations is not None

        print(f"Finding optimal policy using {self._mc_iterations} monte carlo iterations\n")
        action_values = {action: 0 for action in range(self._num_actions)}
        for i in range(self._mc_iterations):
            x = rmvnorm(1, rho=0, dim=self._context_ndim)
            x[:, 0] = 1.00
            for a in range(self._num_actions):
                z = self.get_surrogates(x, [a])
                y = self.get_outcome_noiseless(z)
                action_values[a] += float(y)

        self.true_expected_reward = {action: action_values[action] / self._mc_iterations
                                     for action in action_values.keys()
                                     }
        self.optimal_action = max(
            self.true_expected_reward,
            key=self.true_expected_reward.get
        )

    def gen_sample(self, num_samples: int) -> BanditData:
        # Implemented inefficiently for clarity/debugging
        context = rmvnorm(num_samples, rho=0, dim=self._context_ndim)
        context[:, 0] = 1.0
        action = np.zeros(num_samples, dtype=int)
        propensity = np.zeros(num_samples)
        for i in range(num_samples):
            action[i], propensity[i] = self._pi(context[i, :])
        surrogate = np.zeros((num_samples, self._surrogate_ndim))
        for i in range(num_samples):
            surrogate[i, :] = np.dot(context[i, :], self._theta[action[i]]) + \
                      np.sqrt(self._surrogate_var) * nr.randn(self._surrogate_ndim)

        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            outcome[i] = np.sum(surrogate[i, :]) + self._normal_sd * nr.randn(1)
        return BanditData(
            context=context,
            action=action,
            surrogate=surrogate,
            outcome=outcome,
            propensity=propensity
        )

    def get_context(self, num_samples: int):
        x = rmvnorm(
            num_samples,
            dim=self._context_ndim,
            rho=0
        )
        x[:, 0] = 1.0
        return x

    def get_surrogates(
            self,
            context: np.ndarray,
            action: Union[np.ndarray,
            list]
    ) -> np.ndarray:
        assert context.shape[0] == len(action), \
            "number of treatments differs from number of samples"

        num_samples = context.shape[0]

        surrogate = np.zeros((num_samples, self._surrogate_ndim))

        for i in range(num_samples):
            surrogate[i, :] = np.dot(context[i, :], self._theta[int(action[i])]) + \
                      np.sqrt(self._surrogate_var) * nr.randn(self._surrogate_ndim)

        return surrogate

    def get_outcome(self, surrogate: np.ndarray):
        num_samples = surrogate.shape[0]
        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            outcome[i] = np.sum(surrogate[i, :]) + self._normal_sd * nr.randn(1)
        return outcome

    def get_outcome_noiseless(self, surrogate: np.ndarray):
        num_samples = surrogate.shape[0]
        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            outcome[i] = np.sum(surrogate[i, :])
        return outcome



class TestDeuxGenerativeModel(ContextualBandit):
    def __init__(
            self,
            monotone_func: Callable[[float], float],
            log_normal_mean: float,
            log_normal_var: float,
            normal_var: float,
            autoregressive_cov: float,
            context_ndim: int,
            num_actions: int,
            action_selection: Callable[[np.ndarray], int],
            coefficients: List[np.ndarray],  # one for each action
            surrogate_var: float, # variance for distribution over surrogates
            mc_iterations: int = None
    ) -> None:
        self.optimal_policy = None
        self._monotone_func = monotone_func
        self._log_normal_mean = log_normal_mean
        self._log_normal_var = log_normal_var
        self._log_normal_sd = np.sqrt(log_normal_var)
        self._normal_var = normal_var
        self._normal_sd = np.sqrt(normal_var)
        self._autoregressive_cov = autoregressive_cov
        self._context_ndim = context_ndim
        self._num_actions = num_actions
        self._pi = action_selection
        self._theta = coefficients
        self._surrogate_var = surrogate_var
        self._surrogate_ndim = self._theta[0].shape[1]
        self._mc_iterations = mc_iterations

        assert len(self._theta) == self._num_actions, \
            "dim theta  != num_actions"


    def get_optimal_action(self):
        # Perform MC to estimate optimal action

        assert self._mc_iterations is not None

        print(f"Finding optimal policy using {self._mc_iterations} monte carlo iterations\n")
        action_values = {action: 0 for action in range(self._num_actions)}
        for i in range(self._mc_iterations):
            x = rmvnorm(1, rho=self._autoregressive_cov, dim=self._context_ndim)
            x[:, 0] = 1.0
            for a in range(self._num_actions):
                z = self.get_surrogates(x, [a])
                y = self.get_outcome(z)
                action_values[a] += float(y)

        self.true_expected_reward = {action: action_values[action] / self._mc_iterations
                                     for action in action_values.keys()
                                     }

        self.optimal_action = max(self.true_expected_reward, key=self.true_expected_reward.get)

    def gen_sample(self, num_samples: int) -> BanditData:
        # Implemented inefficiently for clarity/debugging
        context = rmvnorm(num_samples, rho=self._autoregressive_cov, dim=self._context_ndim)
        context[:, 0] = 1.0
        action = np.zeros(num_samples, dtype=int)
        propensity = np.zeros(num_samples)
        for i in range(num_samples):
            action[i], propensity[i] = self._pi(context[i, :])
        surrogate = np.zeros((num_samples, self._surrogate_ndim))
        for i in range(num_samples):
            surrogate[i, :] = np.dot(context[i, :], self._theta[action[i]]) + \
                      np.sqrt(self._surrogate_var) * nr.randn(self._surrogate_ndim)

        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            for j in range(self._surrogate_ndim):
                indicator = float(surrogate[i, j] < 0)
                if j > 1:
                    indicator *= np.prod(
                        surrogate[i, 0:j] >= 0
                    )
                outcome[i] += indicator * (self._monotone_func(norm.cdf(surrogate[i, j])) + (self._surrogate_ndim - j))
            outcome[i] = outcome[i] * np.exp(
                nr.randn(1) * self._log_normal_sd + self._log_normal_mean
            ) + self._normal_sd * nr.randn(1)
        return BanditData(context=context, action=action, surrogate=surrogate, outcome=outcome, propensity=propensity)

    def get_context(self, num_samples: int):
        x = rmvnorm(num_samples, rho=self._autoregressive_cov, dim=self._context_ndim)
        x[:, 0] = 1.0
        return x

    def get_surrogates(self, context: np.ndarray, action: Union[np.ndarray, list]) -> np.ndarray:

        assert context.shape[0] == len(action), "number of treatments differs from number of samples"

        num_samples = context.shape[0]

        surrogate = np.zeros((num_samples, self._surrogate_ndim))

        for i in range(num_samples):
            surrogate[i, :] = np.dot(context[i, :], self._theta[int(action[i])]) + \
                      np.sqrt(self._surrogate_var) * nr.randn(self._surrogate_ndim)

        return surrogate

    def get_outcome(self, surrogate: np.ndarray):

        num_samples = surrogate.shape[0]

        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            for j in range(self._surrogate_ndim):
                indicator = float(surrogate[i, j] < 0)
                if j > 1:
                    indicator *= np.prod(
                        surrogate[i, 0:j] >= 0
                    )
                outcome[i] += indicator * (self._monotone_func(norm.cdf(surrogate[i, j])) + (self._surrogate_ndim - j))
            outcome[i] = outcome[i] * np.exp(
                nr.randn(1) * self._log_normal_sd + self._log_normal_mean
            ) + self._normal_sd * nr.randn(1)

        return outcome


class AveGenerativeModel(ContextualBandit):
    def __init__(
            self,
            monotone_func: Callable[[float], float],
            log_normal_mean: float,
            log_normal_var: float,
            normal_var: float,
            autoregressive_cov: float,
            context_ndim: int,
            num_actions: int,
            action_selection: Callable[[np.ndarray], int],
            coefficients: List[np.ndarray],  # one for each action
            surrogate_var: float, # variance for distribution over surrogates
            mc_iterations: int = None
    ) -> None:
        self.optimal_policy = None
        self._monotone_func = monotone_func
        self._log_normal_mean = log_normal_mean
        self._log_normal_var = log_normal_var
        self._log_normal_sd = np.sqrt(log_normal_var)
        self._normal_var = normal_var
        self._normal_sd = np.sqrt(normal_var)
        self._autoregressive_cov = autoregressive_cov
        self._context_ndim = context_ndim
        self._num_actions = num_actions
        self._pi = action_selection
        self._theta = coefficients
        self._surrogate_var = surrogate_var
        self._surrogate_ndim = self._theta[0].shape[1]
        self._mc_iterations = mc_iterations

        assert len(self._theta) == self._num_actions, \
            "dim theta  != num_actions"


    def get_optimal_action(self):
        # Perform MC to estimate optimal action
        assert self._mc_iterations is not None

        print(f"Finding optimal policy using {self._mc_iterations} monte carlo iterations\n")
        action_values = {action: 0 for action in range(self._num_actions)}
        for i in range(self._mc_iterations):
            x = rmvnorm(1, rho=0, dim=self._context_ndim)
            x[:, 0] = 1.0
            for a in range(self._num_actions):
                z = self.get_surrogates(x, [a])
                y = self.get_outcome_noiseless(z)
                action_values[a] += float(y)

        self.true_expected_reward = {action: action_values[action] / self._mc_iterations
                                     for action in action_values.keys()
                                     }
        self.optimal_action = max(
            self.true_expected_reward,
            key=self.true_expected_reward.get
        )

    def gen_sample(self, num_samples: int) -> BanditData:
        # Implemented inefficiently for clarity/debugging
        context = rmvnorm(num_samples, rho=0, dim=self._context_ndim)
        context[:, 0] = 1.0
        action = np.zeros(num_samples, dtype=int)
        propensity = np.zeros(num_samples)
        for i in range(num_samples):
            action[i], propensity[i] = self._pi(context[i, :])
        surrogate = np.zeros((num_samples, self._surrogate_ndim))
        for i in range(num_samples):
            surrogate[i, :] = np.dot(context[i, :], self._theta[action[i]]) + \
                      np.sqrt(self._surrogate_var) * nr.randn(self._surrogate_ndim)

        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            outcome[i] = (surrogate[i, 0] + surrogate[i, 1] + surrogate[i, 2]/10
                          + self._normal_sd * nr.randn(1))

        return BanditData(
            context=context,
            action=action,
            surrogate=surrogate,
            outcome=outcome,
            propensity=propensity
        )

    def get_context(self, num_samples: int):
        x = rmvnorm(
            num_samples,
            dim=self._context_ndim,
            rho=0
        )
        x[:, 0] = 1.0
        return x

    def get_surrogates(
            self,
            context: np.ndarray,
            action: Union[np.ndarray,
            list]
    ) -> np.ndarray:
        assert context.shape[0] == len(action), \
            "number of treatments differs from number of samples"

        num_samples = context.shape[0]

        surrogate = np.zeros((num_samples, self._surrogate_ndim))

        for i in range(num_samples):
            surrogate[i, :] = np.dot(context[i, :], self._theta[int(action[i])]) + \
                      np.sqrt(self._surrogate_var) * nr.randn(self._surrogate_ndim)

        return surrogate

    def get_outcome(self, surrogate: np.ndarray):
        num_samples = surrogate.shape[0]
        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            outcome[i] = (surrogate[i, 0] + surrogate[i, 1] +
                          surrogate[i, 2]/10 +
                          self._normal_sd * nr.randn(1))
        return outcome

    def get_outcome_noiseless(self, surrogate: np.ndarray):
        num_samples = surrogate.shape[0]
        outcome = np.zeros(num_samples)
        for i in range(num_samples):
            outcome[i] = (surrogate[i, 0] + surrogate[i, 1] +
                          surrogate[i, 2]/10)
        return outcome


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    n = 250
    p = 15
    q = 3
    num_actions = 2
    theta = list()
    theta.append(np.zeros((p, q)))
    theta[0][0:4, 0] = np.array([1, 1, 2, 3])
    theta[0][4:8, 1] = np.array([-1, 1, 0.5, 0.5])
    theta[0][8:12, 2] = np.array([1, 3, -1, 1])
    theta.append(abs(theta[0])*-10 - 10)
    autoregressive_cov = 0.3
    monotone_func = gen_gx()
    pi = gen_pix(num_actions)
    log_normal_mean = 1.0
    log_normal_var = 0.25 ** 2
    normal_var = 10
    surrogate_var = 1.0
    gen_model = CopulaGenerativeModel(
        monotone_func=monotone_func,
        log_normal_mean=log_normal_mean,
        log_normal_var=log_normal_var,
        autoregressive_cov=autoregressive_cov,
        context_ndim=p,
        num_actions=num_actions,
        action_selection=pi,
        coefficients=theta,
        normal_var=normal_var,
        surrogate_var=surrogate_var,
        mc_iterations=100000
    )


    theta_s = np.array([2,1,.5])
    mon_model = MonotoneGenerativeModel(
        log_normal_mean=log_normal_mean,
        log_normal_var=log_normal_var,
        autoregressive_cov=autoregressive_cov,
        context_ndim=p,
        num_actions=num_actions,
        action_selection=pi,
        coefficients_c=theta,
        coefficients_s=theta_s,
        normal_var=normal_var,
        surrogate_var=surrogate_var,
        mc_iterations=100000
    )

    # gen_model.get_optimal_action()

    ## Burn-in
    copula_sample = gen_model.gen_sample(1000)
    mon_sample = mon_model.gen_sample(n)

    plt.scatter(x=mon_sample._surrogate[:,0], y = mon_sample._outcome)
    plt.show()

    plt.scatter(x=mon_sample._surrogate[:,1], y = mon_sample._outcome)
    plt.show()

    plt.scatter(x=mon_sample._surrogate[:,2], y = mon_sample._outcome)
    plt.show()

    # copula
    plt.scatter(x=copula_sample._surrogate[:,0], y = copula_sample._outcome)
    plt.show()

    plt.scatter(x=copula_sample._surrogate[:,1], y = copula_sample._outcome)
    plt.show()

    plt.scatter(x=copula_sample._surrogate[:,2], y = copula_sample._outcome)
    plt.show()

    ## Generate random linear policies
    # num_policies = 1000
    # basket = LinearBasket.generate_random_basket(
    #     num_policies=num_policies,
    #     num_actions=num_actions,
    #     context_ndim=p
    # )
    # beta_ad_hoc = list()
    # beta_ad_hoc.append(np.array(
    #     [1, 1, 2, 3, -1, 1, 0.5, 0.5, 1, 3, -1, 1, 0, 0, 0]
    # ))
    # beta_ad_hoc.append(-beta_ad_hoc[0])
    # basket.add_policy(LinearPolicy(2, beta_ad_hoc))
    # non_dominated_basket = LinearBasket()
    # non_dominated_indices = set(range(len(basket)))
    # for i in range(len(basket)):
    #     i_is_dominated = False
    #     for j in non_dominated_indices:
    #         if i == j:
    #             continue
    #         delta = basket[j].policy_to_measure(init_sample) - \
    #                 basket[i].policy_to_measure(init_sample)
    #         i_is_dominated = gurobi_check_is_dominated(
    #             delta,
    #             init_sample.get_constraint_matrix(),
    #             init_sample.get_constraint_vector()
    #         )
    #         if i_is_dominated:
    #             print(i, " dominated by ", j)
    #             non_dominated_indices.remove(i)
    #             break
    #     if not i_is_dominated:
    #         print(i, " is not dominated")
    # pdb.set_trace()
