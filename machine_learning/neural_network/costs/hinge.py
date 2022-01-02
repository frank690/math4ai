"""This module defines the hinge loss function."""

__all__ = [
    "hinge_loss",
]

import numpy as np


def hinge_loss(h: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the hinge loss for the given hypothesis (h) in contrast to the true results (y).
    :param h: Hypothesis of the NN to compare with y.
    :param y: True results of the data. y is expected to ONLY consist of -1 and +1 !
    :return: Cost/Loss of the current hypothesis.
    """
    return np.mean(np.maximum(0, 1 - h * y))
