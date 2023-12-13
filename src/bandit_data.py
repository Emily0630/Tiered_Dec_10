from __future__ import annotations
from typing import Optional, Callable, NamedTuple
import numpy as np
import numpy.random as nr
from collections import namedtuple
import pdb


class BanditData:
    """
    A class to store data from a contextual bandit with surrogate outcomes.

    We have taken a functional object-oriented approach to this class in that adding
    data creates and returns a new BanditData object. However, when updating we do
    not allow the user to change less_than operator.
    """
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

        # if any of the data is not None, then all must be not be None
        if self._context is not None:
            n = context.shape[0]
            if action.shape[0] != n:
                raise ValueError("action must have same number of rows as context")
            if surrogate.shape[0] != n:
                raise ValueError("surrogate must have same number of rows as context")
            if outcome.shape[0] != n:
                raise ValueError("outcome must have same number of rows as context")
            #if propensity.shape[0] != n:
                #raise ValueError("propensity must have same number of rows as context")

        if less_than is None:
            def z_less_than(z1, z2):
                if np.all(np.less_equal(z1, z2)):
                    return 1
                if np.all(np.greater_equal(z1, z2)):
                    return -1
                return 0
            less_than = z_less_than
        self._less_than = less_than
        
    def i_less_j(self, i: int, j: int) -> int:
        return self._less_than(self._surrogate[i, :], self._surrogate[j, :])

    def add_observations(
            self,
            context_new: np.ndarray,
            action_new: np.ndarray,
            surrogate_new: np.ndarray,
            outcome_new: np.ndarray,
            propensity_new: Optional[np.ndarray] = None
    ) -> BanditData:
        if self._context is None:
            return BanditData(context_new, action_new, surrogate_new, outcome_new, propensity_new, self._less_than)
        return BanditData(
            np.append(self._context, context_new, axis=0),
            np.append(self._action, action_new),
            np.append(self._surrogate, surrogate_new, axis=0),
            np.append(self._outcome, outcome_new),
            np.append(self._propensity, propensity_new),
            self._less_than
        )

    def get_surrogate_dim(self):
        return self._surrogate.shape[1]

    def __len__(self):
        if self._action is not None:
            return len(self._action)
        return 0

    def __getitem__(self, item) -> NamedTuple:
        if item > len(self):
            raise IndexError("Index out of range")
        BanditRow = namedtuple(
            "BanditRow",
            ['context', 'action', 'surrogate', 'outcome', 'propensity']
        )
        return BanditRow(
            self._context[item, :],
            self._action[item],
            self._surrogate[item, :],
            self._outcome[item],
            self._propensity[item]
        )


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


    ## Now let's add data one row at a time
    num_new_data_points = 1000
    for i in range(num_new_data_points):
        x_new = nr.normal(size=(1, p))
        a_new = np.sign(nr.randn(1))
        z_new = x_new @ b
        y_new = a_new*(z_new @ np.ones(k)) + nr.normal(size=1)
        propensity_new = 1/2.0
        bandit_data = bandit_data.add_observations(x_new, a_new, z_new, y_new, propensity_new)
    pdb.set_trace()

















