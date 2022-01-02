"""This module defines the l2 regularization function."""

__all__ = [
    "l2",
]


import numpy as np


def l2(w: np.ndarray, l: float = 0.01, gradient: bool = False) -> np.ndarray:
    """
    Compute the l2 regularization for the given weights (w) and the regularization parameter (l).
    :param l: Regularization parameter.
    :param w: Weight to compute l2 norm of.
    :param gradient: Flag to indicate if gradient should be returned.
    :return: L2 regularization penalty for the given weights.
    """
    if gradient is True:
        return 2 * l * np.sum(w)
    return l * np.sum(np.square(w))
