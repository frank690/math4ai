"""This module contains functionality around the support vector machine (SVM)"""

import numpy as np
from typing import Tuple


class LinearSVM:
    """Class that represents a linear SVM for separable data."""

    def __init__(self):
        """
        store given parameters and initialize the class.
        """
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM to the given data.
        :param X: input data (features) that the SVM should fit. Each row should be a sample. Each column a feature.
        :param y: output data (target) that the SVM should fit.
        """
        self.w = np.random.random((X.shape[1], 1))
        self.b = np.random.random((1, 1))
        self.X = X
        self.y = np.reshape(y, (len(y), 1))

    def cost(self, gradient: bool = False) -> Tuple:
        """
        Defines the cost function.
        :param gradient: Flag to indicate whether cost or its gradient should be returned.
        :return: The current cost of the SVM on the given data or (with the gradient flagged)
        the gradient of the same (with respect to weight and bias).
        """
        if gradient is True:
            mask = np.squeeze(self.y * (self.X @ self.w + self.b) > 1)
            cost_w = np.multiply(self.y, self.X)
            cost_w[mask] = 0
            cost_b = -self.y
            cost_b[mask] = 0
            return np.reshape(
                self.w - np.mean(cost_w, axis=0), self.w.shape
            ), np.reshape(np.mean(cost_b, axis=0), self.b.shape)
        return 0.5 * self.w.T @ self.w + np.mean(
            np.maximum(0, 1 - np.multiply(self.y, (self.X @ self.w + self.b)))
        )

    def gradient_descent(self, batch_size: int = 1):
        """
        Train the SVM via gradient descent with a specific batch size.
        :param batch_size:
        :return:
        """
