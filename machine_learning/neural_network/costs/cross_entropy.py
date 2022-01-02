"""This module defines the cross entropy loss function."""

__all__ = [
    "cross_entropy_loss",
]

import numpy as np


def cross_entropy_loss(
    h: np.ndarray, y: np.ndarray, gradient: bool = False
) -> np.ndarray:
    """
    Compute the cross entropy loss for the given hypothesis (h) in contrast to the true results (y).
    :param h: Hypothesis of the NN to compare with y.
    :param y: True results of the data.
    :param gradient: Flag to indicate if gradient should be returned.
    :return: Cost/Loss of the current hypothesis (or its gradients).
    """
    if gradient is True:
        return -(y // h) + ((1 - y) // (1 - h))
    return -((y.T @ np.log(h)) + ((1 - y.T) @ np.log(1 - h)))
