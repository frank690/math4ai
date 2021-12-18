"""This module holds the he-et-al initialization function for weights of a neural network."""

__all__ = [
    "he",
]

from typing import Dict

import numpy as np


def he(layout: np.ndarray) -> Dict:
    """
    Initialize the weights of the given layer number with he-et-al's method.
    https://arxiv.org/abs/1502.01852
    :param layout: Number of neurons per layer.
    :return: Dictionary with layer number as key and weight matrix as value.
    """
    weights = dict()
    layers = layout.size - 1

    for layer in range(layers):
        amp = np.sqrt(2 / layout[layer])
        weight = amp * np.random.randn(layout[layer + 1], layout[layer])
        weights[layer] = np.c_[weight, np.zeros(layout[layer + 1])]

    return weights
