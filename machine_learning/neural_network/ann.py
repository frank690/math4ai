"""This module holds classes and functions to run artificial neural networks of arbitrary size."""

__all__ = [
    "NeuralNetwork",
]

from typing import Tuple

import numpy as np

from machine_learning.neural_network.activations import sigmoid
from machine_learning.neural_network.costs import cross_entropy_loss
from machine_learning.neural_network.regularizations import l2
from machine_learning.neural_network.weight_initializations import he, xavier


class NeuralNetwork:
    """A class that represents a neural network."""

    def __init__(
        self,
        layout: np.ndarray,
        learning_rate: float = 1.0,
        regularization_parameter: float = 0.0,
        iterations: int = 1000,
    ):
        """
        Initialize the neural network.
        :param layout: the layout of the layers as a 1D vector.
        each number represents the number of neurons for that layer.
        e.g. np.array([3,5,4,1]) would result in the input layer having 3,
        the first hidden layer having 5, the second hidden layer having 4 and the output layer having 1 neuron.
        pay attention to the fact that this also predetermines the shape of the expected in- and output data.
        """
        self.L = layout.size - 1
        self.layout = layout

        self.activate = sigmoid
        self.cost = cross_entropy_loss
        self.regularization = l2
        self.weights, self.biases = he(layout=layout)

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization_parameter = regularization_parameter
        self.threshold = 0.5

        self.z = dict()
        self.a = dict()
        self.dz = dict()
        self.da = dict()
        self.h = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the neural network to the given input data (X) and expected output data (y).
        Run the fitting process over the number of given iterations with the given learning_rate.
        :param X: input data.
        :param y: output data.
        :return: Costs after each iteration.
        """
        costs = []
        for i in range(self.iterations):
            self.forward_propagation(X=X)
            costs.append(self.loss(y=y))
            self.backward_propagation(y=y)
            self.gradient_descent()

        return np.concatenate(costs)

    def loss(self, y: np.ndarray, gradient: bool = False) -> np.ndarray:
        """
        Compute the loss of the current model with respect to the chosen cost and regularization function.
        :param y: output data.
        :param gradient: Flag to indicate if gradient should be returned.
        :return: Loss of the current model.
        """
        cost = self.cost(h=self.h, y=y, gradient=gradient)
        cost += np.sum(
            [
                self.regularization(
                    w=w, gradient=gradient, l=self.regularization_parameter
                )
                for w in self.weights.values()
            ]
        )
        return cost

    def forward_propagation(self, X: np.ndarray):
        """
        Run the forward propagation from input to output layer.
        """
        self.a[0] = X

        for l in range(1, self.L + 1):
            self.z[l] = (self.a[l - 1] @ self.weights[l]) + self.biases[l]
            self.a[l] = self.activate(self.z[l])
        self.h = self.a[self.L]

    def backward_propagation(self, y: np.ndarray):
        """
        Run the forward propagation from input to output layer.
        :param y: output data to run backwards through the neural network
        """
        self.da[self.L] = self.loss(y=y, gradient=True)

        for l in range(self.L, 0, -1):
            self.dz[l] = self.activate(data=self.z[l], gradient=True)
            self.da[l - 1] = self.dz[l] @ self.weights[l].T

    def gradient_descent(self):
        """
        Run the gradient descent method to update the weights and biases.
        """
        for l in range(1, self.L + 1):
            tri = (self.da[l] * self.dz[l]) / (self.da[l].shape[0])
            self.weights[l] -= self.learning_rate * (self.a[l - 1].T @ tri)
            self.biases[l] -= self.learning_rate * np.sum(tri, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the result, given some data.
        :param X: The data to make predictions on.
        :return: the computed result.
        """
        self.forward_propagation(X=X)
        return self.h

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """
        Make a binary class decision on the given input data (X).
        :param X: Data to predict.
        :return: Predictions as binary result (1 or 0).
        """
        y_pred = self.predict(X=X)
        return np.array([[1 if yi[0] >= self.threshold else 0 for yi in y_pred]]).T

    def accuracy(self, X: np.ndarray, y: np.ndarray):
        """
        Predict the outcomes of the given data (X) and compare it with the true outcome (y).
        Compute the accuary and return that value.
        :param X: input data.
        :param y: output data.
        """
        y_pred = self.predict_class(X)

        TP = np.sum((y == 1) & (y_pred == 1))
        TN = np.sum((y == 0) & (y_pred == 0))
        FP = np.sum((y == 0) & (y_pred == 1))
        FN = np.sum((y == 1) & (y_pred == 0))

        print(
            f"True Positives: {TP}\nFalse Positives: {FP}\nFalse Negatives: {FN}\nTrue Negatives: {TN}\nAccurary: "
            f"{np.round((TP + TN) * 100 / (FP + FN + TP + TN), 4)}%"
        )


def generate_data(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a data for training our linear model.
    :param N: number of samples multiplier.
    :return: tuple of x and y data as numpy ndarrays.
    """
    X = np.repeat(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), N, axis=0)
    X = X + np.random.randn(4 * N, 2) * 0.2
    y = np.repeat([1, 0, 0, 1], N)
    y = np.reshape(y, (len(y), 1))

    return X, y


if __name__ == "__main__":
    X_train, y_train = generate_data(N=100)
    X_test, y_test = generate_data(N=50)

    nn = NeuralNetwork(layout=np.array([2, 3, 1]))
    J = nn.fit(X=X_train, y=y_train)

    nn.accuracy(X=X_test, y=y_test)
