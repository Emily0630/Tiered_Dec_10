from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np
import numpy.random as nr
import pdb
from src.bandit_data import BanditData
from scipy.stats.qmc import LatinHypercube

class DeterministicPolicy(ABC):
    @abstractmethod
    def decision(self, x: np.ndarray) -> int:
        pass

    @abstractmethod
    def policy_to_measure(self, data: BanditData) -> np.ndarray:
        pass


class LinearPolicy(DeterministicPolicy):
    def __init__(self, num_actions: int, beta: List[np.ndarray]) -> None:
        self._num_actions = num_actions
        self._beta = beta

    def decision(self, x: np.ndarray) -> int:
        best_action = 0
        best_score = np.dot(x, self._beta[0])
        for j in range(1, self._num_actions):
            score = np.dot(x, self._beta[j])
            if score > best_score:
                best_action = j
                best_score = score
        return best_action

    def policy_to_measure(self, data: BanditData) -> np.ndarray:
        measure = np.zeros(len(data))
        for i, (x, a, z, y, p) in enumerate(data):
            if a == self.decision(x):
                measure[i] = 1/p
        measure /= np.sum(measure)
        return measure


class LinearBasket:
    def __init__(self, basket: Optional[List[LinearPolicy]] = None) -> None:
        self._basket = basket

    def add_policy(self, new_policy: LinearPolicy) -> LinearBasket:
        if self._basket is None:
            return LinearBasket([new_policy])
        return LinearBasket(self._basket + [new_policy])

    def __len__(self):
        if self._basket is None:
            return 0
        return len(self._basket)

    def __getitem__(self, item):
        if item > len(self):
            raise IndexError("Index out of range")
        return self._basket[item]

    @staticmethod
    def generate_random_basket(
            num_policies: int,
            num_actions: int,
            x_dim: int
    ) -> LinearBasket:
        basket = LinearBasket()
        lhs = LatinHypercube(x_dim)
        all_beta = lhs.random(n=num_policies*num_actions) - 1/2
        for j in range(num_policies):
            beta = []
            for k in range(num_actions):
                beta += [all_beta[j*num_actions + k, ]]
            beta_policy = LinearPolicy(num_actions=num_actions, beta=beta)
            basket = basket.add_policy(beta_policy)
        return basket


class LinearBasketBinary:
    def __init__(self, basket: Optional[List[LinearPolicy]] = None) -> None:
        self._basket = basket

    def add_policy(self, new_policy: LinearPolicy) -> LinearBasket:
        if self._basket is None:
            return LinearBasket([new_policy])
        return LinearBasket(self._basket + [new_policy])

    def __len__(self):
        if self._basket is None:
            return 0
        return len(self._basket)

    def __getitem__(self, item):
        if item > len(self):
            raise IndexError("Index out of range")
        return self._basket[item]

    @staticmethod
    def generate_random_basket(
            num_policies: int,
            x_dim: int
    ) -> LinearBasket:
        basket = LinearBasket()
        lhs = LatinHypercube(x_dim)
        all_beta = lhs.random(n=num_policies) - 1/2
        for j in range(num_policies):
            beta = []
            beta.append(all_beta[j])
            beta.append(-all_beta[j])
            beta_policy = LinearPolicy(num_actions=2, beta=beta)
            basket = basket.add_policy(beta_policy)

            # Now add opposite
            beta = []
            beta.append(-all_beta[j])
            beta.append(all_beta[j])
            beta_policy = LinearPolicy(num_actions=2, beta=beta)
            basket = basket.add_policy(beta_policy)

        # Add constant policies
        beta = []
        beta.append(np.ones(x_dim))
        beta.append(-np.ones(x_dim))
        beta_policy = LinearPolicy(num_actions=2, beta=beta)
        basket = basket.add_policy(beta_policy)

        beta = []
        beta.append(-np.ones(x_dim))
        beta.append(np.ones(x_dim))
        beta_policy = LinearPolicy(num_actions=2, beta=beta)
        basket = basket.add_policy(beta_policy)

        # Intercept only
        e1 = np.zeros(x_dim)
        e1[0] = 1.0
        beta = []
        beta.append(e1)
        beta.append(-e1)
        beta_policy = LinearPolicy(num_actions=2, beta=beta)
        basket = basket.add_policy(beta_policy)

        beta = []
        beta.append(-e1)
        beta.append(e1)
        beta_policy = LinearPolicy(num_actions=2, beta=beta)
        basket = basket.add_policy(beta_policy)

        return basket


if __name__ == "__main__":
    num_policies = 1000
    num_actions = 3
    x_dim = 15
    random_basket = LinearBasket.generate_random_basket(
        num_policies=num_policies,
        num_actions=num_actions,
        x_dim=x_dim
    )
    pdb.set_trace()
