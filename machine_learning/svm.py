"""This module contains functionality around the support vector machine (SVM)"""

import numpy as np
from random import Random
from typing import Tuple, List, Optional


class LinearSVM:
    """Class that represents a linear SVM."""
    def __init__(self, C: float = 1):
        """
        store given parameters and initialize the class.
        :param C: regularization parameter.
        """
        self.C = C
        self.w = None
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM to the given data.
        :param X: input data (features) that the SVM should fit. Each row should be a sample. Each column a feature.
        :param y: output data (target) that the SVM should fit.
        """
        self.w = np.random.random((X.shape[1], 1))

    def cost(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Defines the cost function.
        :param X: input data (features) that the SVM should fit. Each row should be a sample. Each column a feature.
        :param y: output data (target) that the SVM should fit.
        :return: resulting cost as (1x1) array.
        """
        n = X.shape[0]
        zero_vector = np.zeros((n, 1))

        return 0.5 * (self.w.T @ self.w) + self.C * (
            (1/n) * np.sum(np.maximum(zero_vector, 1 - (y * ((X @ self.w) + self.b))))
        )


if __name__ == '__main__':
    pass
