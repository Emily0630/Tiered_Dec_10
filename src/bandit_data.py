from __future__ import annotations

import scipy.interpolate
from scipy.sparse import dok_matrix  # was dok_array but not available in scipy (in my system)

# Local package
from src.monotonic_tree import *


class BanditData:
    def __init__(
            self,
            context: Optional[np.ndarray] = None,
            action: Optional[np.ndarray] = None,
            surrogate: Optional[np.ndarray] = None,
            outcome: Optional[np.ndarray] = None,
            propensity: Optional[np.ndarray] = None,
            less_than: Optional[Callable] = None
    ) -> None:
        self._context = context
        self._action = action
        self._surrogate = surrogate
        self._outcome = outcome
        self._propensity = propensity
        self._constraint_vector = None
        self._constraint_matrix = None
        if less_than is None:
            def z_less_than(z1, z2):
                if np.all(np.less_equal(z1, z2)):
                    return 1  # z1 <= z2
                if np.all(np.greater_equal(z1, z2)):
                    return -1  # z1 > z2
                return 0  # not comparable

            less_than = z_less_than
        self._less_than = less_than

    def add_observations(
            self,
            context_new: np.ndarray,
            action_new: np.ndarray,
            surrogate_new: np.ndarray,
            outcome_new: np.ndarray,
            propensity_new: Optional[np.ndarray] = None
    ) -> BanditData:
        if self._context is None:
            return BanditData(context_new, action_new, surrogate_new, outcome_new, propensity_new)
        return BanditData(
            np.append(self._context, context_new, axis=0),
            np.append(self._action, action_new),
            np.append(self._surrogate, surrogate_new, axis=0),
            np.append(self._outcome, outcome_new),
            np.append(self._propensity, propensity_new),
            self._less_than
        )

    def get_constraint_matrix(self):
        return self._constraint_matrix

    def get_constraint_vector(self):
        return self._constraint_vector

    def __getitem__(self, item):
        if item > len(self):
            raise IndexError("Index out of range")
        return (
            self._context[item, :],
            self._action[item],
            self._surrogate[item, :],
            self._outcome[item],
            self._propensity[item]
        )

    def __len__(self):
        if self._action is None:
            return 0
        return len(self._action)

