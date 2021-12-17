"""This module holds classes and functions to run artificial neural networks of arbitrary size."""

__all__ = [
    "NeuralNetwork",
]

import numpy as np


class NeuralNetwork:
    """A class that represents a neural network."""
    def __init__(self, layout: np.ndarray, initialization: str = "he"):
        """
        Initialize the neural network.
        :param layout: the layout of the layers as a 1D vector.
        each number represents the number of neurons for that layer.
        e.g. np.array([3,5,4,1]) would result in the input layer having 3,
        the first hidden layer having 5, the second hidden layer having 4 and the output layer having 1 neuron.
        pay attention to the fact that this also predetermines the shape of the expected in- and output data.
        :param initialization: names the method of weight initialization that should be used.
        valid values are: xavier, he
        """
        self.layout = layout
        self.initialization = initialization

        self.weights = dict()

        self._construct()

    def _construct(self):
        """
        Constructs the neural networks by initializing the weights and biases given the layout parameter.
        """
        layers = self.layout.size - 1
        for layer in range(layers):
            self.weights[layer] = self._initialize_weights(layer=layer)

    def _initialize_weights(self, layer: int) -> float:
        """
        Compute the amplification that the normal distribution of the weights/neurons will be multiplied with
        in order to achieve a good behaving neural network from the get-go.
        Different approaches can be chosen upon initializing this class itself via the "initialization" parameter.
        :param layer: Number of current layer.
        :return: Amplification factor for the normal distribution.
        """
        if self.initialization == "xavier":
            return self._init_xavier(layer=layer)
        elif self.initialization == "he":
            return self._init_he(layer=layer)
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}. Please choose a valid method.")

    def _init_xavier(self, layer: int) -> float:
        """
        Initialize the weights of the given layer number with xavier's method.
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        :param layer: Number of current layer.
        :return: Spread factor for the normal distribution.
        """
        amp = np.sqrt(6) / np.sqrt(self.layout[layer] + self.layout[layer + 1])
        weights = np.random.uniform(-amp, amp, (self.layout[layer + 1], self.layout[layer]))
        return np.c_[weights, np.zeros(self.layout[layer + 1])]

    def _init_he(self, layer: int) -> float:
        """
        Initialize the weights of the given layer number with he-et-al's method.
        https://arxiv.org/abs/1502.01852
        :param layer: Number of current layer.
        :return: Spread factor for the normal distribution.
        """
        amp = np.sqrt(2 / self.layout[layer])
        weights = amp * np.random.randn(self.layout[layer + 1], self.layout[layer])
        return np.c_[weights, np.zeros(self.layout[layer + 1])]
