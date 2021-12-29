"""This module holds the he-et-al initialization function for weights of a neural network."""

__all__ = [
    "he",
]

from typing import Dict, Tuple

import numpy as np


def he(layout: np.ndarray) -> Tuple[Dict, Dict]:
    """
    Initialize the weights and biases of the given layer number with he-et-al's method.
    https://arxiv.org/abs/1502.01852
    :param layout: Number of neurons per layer.
    :return: Tuple of two dictionaries. one of weight matrix and bias vector per number of connected layers.
    """
    weights = dict()
    biases = dict()
    layers = layout.size

    for l in range(1, layers):
        amp = np.sqrt(2 / layout[l - 1])
        weight = amp * np.random.randn(layout[l - 1], layout[l])

        weights[l] = weight
        biases[l] = np.zeros(layout[l])

    return weights, biases
