"""This modules holds functions and classes to generate toy data."""

__all__ = [
    "xnor",
]


from typing import Tuple

import numpy as np


def xnor(N: int, o: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate xnor binary classification data.
    :param N: number of samples multiplier.
    :param o: Overlapping scale. This value should be in the interval (0, 1). A higher value results in greater overlap.
    :return: tuple of x and y data as numpy ndarrays.
    """
    X = np.repeat(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), N, axis=0)
    X = X + np.random.randn(4 * N, 2) * o
    y = np.repeat([1, 0, 0, 1], N)
    y = np.reshape(y, (len(y), 1))

    return X, y
