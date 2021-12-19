"""This module holds classes and functions to run artificial neural networks of arbitrary size."""

__all__ = [
    "NeuralNetwork",
]

import numpy as np

from .activations import sigmoid
from .costs import cross_entropy_loss
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
        self.activate = sigmoid
        self.cost = cross_entropy_loss
        self.parameters = PropagationParameters(layout=layout)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        iterations: int = 100,
        learning_rate: float = 0.1,
    ):
        """
        Fit the neural network to the given input data (X) and expected output data (y).
        Run the fitting process over the number of given iterations with the given learning_rate.
        :param X: input data.
        :param y: output data.
        :param iterations: number of iterations to fit the model.
        :param learning_rate: learning rate for updating the weights.
        """
        self.forward_propagation(a=X)
        # J = self.cost(h=h, y=y)

    def forward_propagation(self, a: np.ndarray):
        """
        Run the forward propagation from input to output layer.
        :param a: input data to run forward through the neural network
        """
        for layer, weight in self.weights.items():
            a = np.column_stack([a, np.ones(a.shape[0])])
            z = a @ weight
            a = self.activate(z)

            self.z[layer] = z
            self.a[layer] = a

    def backward_propagation(self, y: np.ndarray):
        """
        Run the forward propagation from input to output layer.
        :param y: output data to run backwards through the neural network
        """
        for layer, weight in self.weights.items():
            pass


class PropagationParameters:
    """Class to keep track of propagation parameters."""

    def __init__(self, layout: np.ndarray):
        """Initialize the class"""
        self.weights = he(layout=layout)
        self.z = dict()
        self.a = dict()
        self.d = dict()
        self.g = dict()
