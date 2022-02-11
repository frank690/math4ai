"""This module provides several information criterions for model selection"""

__all__ = [
    "bic",
]

import numpy as np


def bic(e: float, k: int, n: int) -> float:
    """
    The bayesian information criterion for evaluating your model selection.
    :param e: Error of the likelihood function of your model.
    :param k: Number of parameters of your model.
    :param n: Number of samples you used to train your model.
    :return: BIC value
    """
    return k * np.log(n) - 2 * np.log(e)
