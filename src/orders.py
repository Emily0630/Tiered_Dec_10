import numpy as np


def product_order(z1, z2):
    if np.all(np.less_equal(z1, z2)):
        return 1  # z1 <= z2
    if np.all(np.greater_equal(z1, z2)):
        return -1  # z1 > z2
    return 0  # not comparable


def no_order(z1, z2):
    return 0  # not comparable
