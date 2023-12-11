import random

import numpy as np
import numpy.random as npr
import pandas as pd
from pyparsing import col
import yaml
import os

from typing import List, Dict, Union, Callable, Any


def _parse_condition(node: Dict):
    """
    Parses conditon dictionary and returns tuple used for evaluating rule on data
    :param node: Dictionary representing a decision rule: contains threshold, variable name or index, operation (leq or geq)
    :return: Tuple
    """
    attrs = list(node.keys())

    assert "threshold" in attrs and "relation" in attrs, "Missing or mispelled fields 'threshold' and 'relation'"
    assert "name" in attrs or "index" in attrs, "Missing or mispelled fields 'threshold' and 'relation'"
    assert "leq" in node['relation'] or "geq" in node['relation'], "'relation' field must include 'geq' or 'leq'"

    assert type(node["threshold"]) == int or type(node["threshold"]) == float, "theshold value mus be int or float"

    operation = -1 if node["relation"] == "leq" else 1

    column_id = node["index"] if "index" in attrs else node["name"]

    return (column_id, node["threshold"], operation)


def _parse_clause(node: Dict):
    """
    :param node: Dictionary representing clause element
    :return: dictionary containing values to evaluate clause against data
    """
    attrs = list(node.keys())
    fields = ["conditions", "agg", "actions", "probs", "adaptive"]
    assert all((elt in attrs for elt in fields)), \
        "clause is missing at least one of the following fields: {}".format(", ".join(fields))

    if not node["agg"] is None:
        assert "and" in node["agg"] or "or" in node["agg"], "agg field must contain and's or or's or be empty"

    conditions = [_parse_condition(node["conditions"][condition]) for condition in node["conditions"]]

    extract = dict(conditions=conditions, agg=node["agg"], actions=node['actions'], probs=node['probs'],
                   adaptive=node['adaptive'])

    return extract


def _parse_rules(tree: Dict):
    """
    Parses clauses from warm start yaml file so they can be evaluated against data.

    :param tree: Dictionary read from warm start yaml files
    :return: List of clauses
    """
    clauses = list(filter(lambda x: 'clause' in x, tree.keys()))
    rules = [_parse_clause(tree[clause]) for clause in clauses]

    return rules


def _eval_condition(X, tau: Union[int, float], b: int):
    """
    -1*X >= -1*tau <=> X <= tau

    Thus -1 checks if X <= tau and 1 checks if X >= tau

    For X binary:
        - To check X == 1 use b = 1 and tau = 1, which yields
            X >= 1,
          which holds only if X = 1, for binary variables
        - To check X == 0 use b = -1 and tau = 0, which yields
            -X >= 0
          which only holds if X == 0 for binary X.

    :param X: variable value
    :param b: relation (-1 = leq, 1 = geq)
    :param tau: threshold
    :return: Boolean

    X = 5, b = -1, tau = 4 will return -5 > -4 which is false
    """
    assert b == -1 or b == 1

    return b * X >= b * tau


def _identity(x):
    return x


def _neg(x):
    return not x


def _or_and(A: bool, B: bool, d: Callable[[Union[int, float]], bool]) -> bool:
    """
    Returns A and B if d = _identity
    Returns A or B if d = _neg

    :param A: boolean
    :param B: boolean
    :param d: identity or negation
    :return: boolean
    """

    return d(d(A) & d(B))


def _eval_clause(
        clause: Dict,
        data: np.ndarray,
        col_id: Callable = _identity) -> Union[List[bool], bool]:
    """
    Evaluates parsed clause against data.

    :param clause: parsed clauses returned from _parse_clause
    :param data:
    :param col_id:
    :return:
    """
    attrs = list(clause.keys())

    assert 'conditions' in attrs and 'agg' in attrs and 'actions' in attrs, "Clause is missing fields: conditions, agg, and actions"

    operation = {'and': _identity, 'or': _neg}

    if len(clause['conditions']) == 1:
        rule = clause['conditions'][0]
        truth_value = _eval_condition(data[:, col_id(rule[0])], rule[1], rule[2])
        return truth_value
    else:

        aggs = clause['agg']
        rules = [_eval_condition(data[:, col_id(rule[0])], rule[1], rule[2]) for rule in clause['conditions']]

        n = len(rules)

        if n == 2:
            truth_values = _or_and(rules[0], rules[1], operation[aggs[0]])
            return truth_values
        else:
            truth_values = _or_and(rules[0], rules[1], operation[aggs[0]])
            for i in range(1, n - 1):
                truth_values = _or_and(truth_values, rules[i + 1], operation[aggs[i]])

            return truth_values


def _to_simplex(data: np.ndarray, actions: list, policy: Callable):
    opt_act = policy(data) 
    return (np.array(actions) == opt_act).astype(float)


def _normalized(values: Union[np.ndarray, list]) -> np.ndarray:
    """Normalize array of values so that they sum to 1

    Args:
        values (Union[np.ndarray, list]): array of nonnegative values

    Returns:
        np.ndarray: array of floats
    """
    values = np.array(values)
    assert all(values >= 0), "values must be nonnegative"

    return values / sum(values)


class WarmStart:
    """
    Class for processing warm start rules and evaluating them against data.
    """

    def __init__(self, param_dict: Dict):

        self._all_actions = param_dict['actions']
        self._clauses = [clause for clause in _parse_rules(param_dict)]
        self._actions = [clause['actions'] for clause in self._clauses]
        self._probs = [_normalized(clause['probs']) for clause in self._clauses]
        self._adaptive = [clause['adaptive'] for clause in self._clauses]
        self._default_actions = param_dict['default']['actions']
        self._default_probs = _normalized(param_dict['default']['probs'])

    def warm_start(self, data):
        """Assign actions according to given warm start rules.

        Args:
            data (_type_): Initial contexts.

        Returns:
            _type_: actions assigned according to warm start rules
        """

        if isinstance(data, pd.DataFrame):
            columns = list(data.columns)

            def _col_id(name: str):
                "Used to subset numpy array by variable name"
                col_id = columns.index(name)
                return col_id

            data = data.to_numpy()

        n = data.shape[0]
        actions = npr.choice(self._default_actions, size=n, p=self._default_probs)
        nest_clause = np.zeros(n).astype(bool)

        num_clauses = len(self._clauses)

        prev_clause = _eval_clause(self._clauses[0], data, col_id=_col_id)

        if sum(prev_clause) > 0:
            actions[prev_clause] = npr.choice(self._actions[0], size=sum(prev_clause), p=self._probs[0])

        for i in range(1, num_clauses):
            curr_clause = _eval_clause(self._clauses[i], data, col_id=_col_id)

            nest_clause = (~ prev_clause) & curr_clause

            if sum(nest_clause) > 0:
                actions[nest_clause] = npr.choice(self._actions[0], size=sum(nest_clause), p=self._probs[0])

                prev_clause[~prev_clause] = curr_clause[~prev_clause]

        assert all(elt in self._default_actions for elt in actions[~prev_clause])

        return actions

    def pick_action(self, data, alpha: float = None, policy: Callable = None):
        """Integrates policy learning from RL algorithm with warm start rules to adaptively change the probability 
        that an action is assinged.

        Args:
            data (_type_): contex
            alpha (float, optional): weight for convext combination. Defaults to None.
            policy (Callable, optional): Policy that maps context to probability distribution over actions. Defaults to None.

        Returns:
            _type_: Assgined actions
        """

        if isinstance(data, pd.DataFrame):
            columns = list(data.columns)

            def _col_id(name: str):
                "Used to subset numpy array by variable name"
                col_id = columns.index(name)
                return col_id

            data = data.to_numpy()
    

        assert not ((alpha is None) ^ (policy is None)), "Both alpha and policy must be speficied together"

        n = data.shape[0]
        # a = random.choices(self._default_actions, weights=self._default_probs, k=n)
        actions = npr.choice(self._default_actions, size=n, p=self._default_probs).astype('<U16')
        nest_clause = np.zeros(n).astype(bool)

        num_clauses = len(self._clauses)

        prev_clause = _eval_clause(self._clauses[0], data, col_id=_col_id)

        if sum(prev_clause) > 0:
            for j in np.where(prev_clause):

                for idx, a in enumerate(self._actions[0]):
                    new_pi = policy(data[j,:]) # maps context to probabilities of actions, new_pi is a dictionary
                    if new_pi[a] == 0 or self._adaptive[0][idx] == 0:
                        continue
                    else:
                        # New action probabilities are convex combination
                        # of previous probabilities and new probabilites
                        self._probs[0][idx] = alpha * self._probs[0][idx] + (1 - alpha) * new_pi[a]
                # normalize action probatilites
                self._probs[0][idx] = _normalized(self._probs[0][idx])
                actions[j] = npr.choice(self._actions[0], size=1, p=self._probs[0])

        for i in range(1, num_clauses):
            curr_clause = _eval_clause(self._clauses[i], data, col_id=_col_id)

            nest_clause = (~ prev_clause) & curr_clause

            if sum(nest_clause) > 0:
                for j in np.where(nest_clause):
                    for idx, a in enumerate(self._actions[i]):
                        new_pi = policy(data[j,:]) # maps context to probabilities of actions, new_pi is a dictionary
                        if new_pi[a] == 0 or self._adaptive[i][idx] == 0:
                            continue
                        else:
                            # New action probabilities are convex combination
                            # of previous probabilities and new probabilites
                            self._probs[i][idx] = alpha * self._probs[i][idx] + (1 - alpha) * new_pi[a]
                    # normalize action probatilites
                    self._probs[i][idx] = _normalized(self._probs[i][idx])
                    actions[j] = npr.choice(self._actions[i], size=1, p=self._probs[i])

                prev_clause[~prev_clause] = curr_clause[~prev_clause]

        assert all(elt in self._default_actions for elt in actions[~prev_clause])

        return actions


if __name__ == '__main__':
    import numpy as np
    import numpy.random as npr
    import pandas as pd
    import yaml

    n = 100

    # df = pd.DataFrame(
    #     dict(tracked_today=npr.choice(range(2), n), cal_today=npr.randn(n) * 200 + 700)
    # )

    filepath = './data/adapt_warm_start.yml'

    with open(filepath) as f:
        warm_start_rules = yaml.safe_load(f)


    variables = []

    clauses = [key for key in warm_start_rules.keys() if 'clause' in key]

    for clause in clauses:
        for condition in warm_start_rules[clause]['conditions']:
            variables.append(warm_start_rules[clause]['conditions'][condition]['name'])

    variables = np.unique(variables)
    print(variables)

    df = pd.DataFrame(np.zeros((n, len(variables))), columns=variables)
    continuous = ['weight_chg_overall', 'cal_today', 'current_week']
    df['cal_today'] =np.random.rand(n)*200 + 1500
    df['weight_chg_overall'] =np.random.rand(n)*200 
    df['current_week'] = np.random.randint(low=0, high=10, size=n)
    for var in variables:
        if var not in continuous:
            df[var] = np.random.randint(low=0, high=2, size=n)

    warm_start = WarmStart(
        param_dict=warm_start_rules
    )

    actions: str = warm_start.pick_action(df)

    print(actions)

# Verifying rules work with smaller file
filepath = './data/warm_start_trunc.yml'

with open(filepath) as f:
    warm_start_short = yaml.safe_load(f)


warm_start_shrt = WarmStart(
        param_dict=warm_start_short
    )


df.loc[0, "current_week"] = 1
df.loc[0, "cal_track_complete_today"] = 1
df.loc[0, "cal_goal_met_today"] = 1
df.loc[0, "am_goal_met_today"] = 1
df.loc[0, "weigh_today"] = 1
df.loc[0, "weight_chg_overall"] = -0.2

actions = warm_start_shrt.pick_action(df)

actions[0]
df.iloc[0,:]