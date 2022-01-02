"""This module holds the xavier initialization function for weights of a neural network."""

__all__ = [
    "xavier",
]

from typing import Dict, Tuple

import numpy as np


def xavier(layout: np.ndarray) -> Tuple[Dict, Dict]:
    """
    Initialize the weights of all layers given their layout with the xavier method.
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    :param layout: Number of neurons per layer.
    :return: Dictionary with layer number as key and weight matrix as value.
    """
    weights = dict()
    biases = dict()
    layers = layout.size

    for l in range(1, layers):
        amp = np.sqrt(6) / np.sqrt(layout[l] + layout[l - 1])
        weight = np.random.uniform(-amp, amp, (layout[l - 1], layout[l]))

        weights[l] = weight
        biases[l] = np.zeros(layout[l])

    return weights, biases
