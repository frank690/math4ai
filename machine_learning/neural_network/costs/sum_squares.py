"""This module defines the sum of squares loss function."""

__all__ = [
    "sum_squares_loss",
]


import numpy as np


def sum_squares_loss(
    h: np.ndarray, y: np.ndarray, gradient: bool = False
) -> np.ndarray:
    """
    Compute the sum of squares loss for the given hypothesis (h) in contrast to the true results (y).
    :param h: Hypothesis of the NN to compare with y.
    :param y: True results of the data.
    :param gradient: Flag to indicate if gradient should be returned.
    :return: Cost/Loss of the current hypothesis (or its gradients).
    """
    if gradient is True:
        return 2 * (h - y)
    return np.sum(np.square(h - y), keepdims=True)
