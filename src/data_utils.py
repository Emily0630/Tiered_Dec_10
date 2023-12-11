import numpy.random as npr
import numpy as np

def r_unit_ball(n: int, d: int, seed: int = 123) -> np.array:
    """
    Generate points uniformly inside a d-dimensional unit ball
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html

    :param n: number of observations
    :param d: dimension
    :param seed: seed for random number generator
    :return: n d-dimensional points sampled uniformly from unit ball.
    """
    rng = npr.default_rng(seed)
    S = rng.normal(size=(n, d))
    U_d = rng.uniform(size=(n,1)) ** (1 / d)
    d_ball = U_d * np.divide(S, np.sqrt((S ** 2).sum(axis=1)).reshape(-1, 1))

    return d_ball
