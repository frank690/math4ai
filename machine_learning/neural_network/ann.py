"""This module holds classes and functions to run artificial neural networks of arbitrary size."""

__all__ = [
    "NeuralNetwork",
]

import numpy as np

from .activations import sigmoid
from .weight_initializations import he, xavier


class NeuralNetwork:
    """A class that represents a neural network."""

    def __init__(self, layout: np.ndarray):
        """
        Initialize the neural network.
        :param layout: the layout of the layers as a 1D vector.
        each number represents the number of neurons for that layer.
        e.g. np.array([3,5,4,1]) would result in the input layer having 3,
        the first hidden layer having 5, the second hidden layer having 4 and the output layer having 1 neuron.
        pay attention to the fact that this also predetermines the shape of the expected in- and output data.
        """
        self.layout = layout
        self.weights = he(layout=layout)
        self.activate = sigmoid
