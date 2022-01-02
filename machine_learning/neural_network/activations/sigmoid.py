"""This module holds the sigmoid activation function for neural networks."""

__all__ = [
    "sigmoid",
]

import numpy as np


def sigmoid(data: np.ndarray, gradient: bool = False) -> np.ndarray:
    """
    Run the sigmoid activation function on the given data and return the result.
    If gradient flag is true, return the gradient of the sigmoid for the given data instead.
    :rtype: object
    :param data: Data to apply sigmoid (or its gradient) to.
    :param gradient: Flag to indicate if gradient should be returned.
    :return: Sigmoid (or its gradient) applied to the given data.
    """
    if gradient is True:
        result = sigmoid(data=data)
        return result * (1.0 - result)

    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-data))
