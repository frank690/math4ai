"""This module defines the cross entropy loss function."""

__all__ = [
    "cross_entropy_loss",
]

import numpy as np


def cross_entropy_loss(h: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the cross entropy loss for the given hypothesis (h) in contrast to the true results (y).
    :param h: Hypothesis of the NN to compare with y.
    :param y: True results of the data.
    :return: Cost/Loss of the current hypothesis.
    """
    return (1 / y.size) * ((y.T @ np.log(h)) + ((1 - y.T) @ np.log(h)))
