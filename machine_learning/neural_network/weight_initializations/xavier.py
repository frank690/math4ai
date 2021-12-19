"""This module holds the xavier initialization function for weights of a neural network."""

__all__ = [
    "xavier",
]

from typing import Dict

import numpy as np


def xavier(layout: np.ndarray) -> Dict:
    """
    Initialize the weights of all layers given their layout with the xavier method.
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    :param layout: Number of neurons per layer.
    :return: Dictionary with layer number as key and weight matrix as value.
    """
    weights = dict()
    layers = layout.size - 1

    for layer in range(layers):
        amp = np.sqrt(6) / np.sqrt(layout[layer] + layout[layer + 1])
        weight = np.random.uniform(-amp, amp, (layout[layer], layout[layer + 1]))
        weights[layer] = np.vstack([weight, np.zeros(layout[layer + 1])])

    return weights
