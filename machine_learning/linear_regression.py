"""This module provides some classes to do linear regression"""

__all__ = [
    "Polynoms",
    "Polynom",
]

from typing import List

import numpy as np


class Polynoms:
    """A class that creates many Polynom classes of different dimensions."""

    def __init__(self, max_dimensions: int, X: np.ndarray, y: np.ndarray):
        """Initialize class by generating a number of Polynom classes."""
        self.polynoms = {}
        self.create_polynoms(max_dimensions=max_dimensions, X=X, y=y)

    def __getitem__(self, index) -> "Polynom":
        """Magic method to return Polynom class when this class is called."""
        return self.polynoms[index]

    def create_polynoms(self, max_dimensions: int, X: np.ndarray, y: np.ndarray):
        """Creates multiple polynoms depending on the max_dimensions parameter."""
        for dimension in range(0, max_dimensions):
            self.create_polynom(dimension, X, y)

    def create_polynom(self, dimension: int, X: np.ndarray, y: np.ndarray):
        """Create polynom and add it to the dictionary of polynoms."""
        self.polynoms[dimension] = Polynom(dimension=dimension, X=X, y=y)

    def mse(self, X: np.ndarray, y: np.ndarray) -> List:
        """Compute mean squared error (on given data) for each Polynomial class that was previously created."""
        mses = []
        for polynom in self.polynoms.values():
            mses.append(polynom.mse(X, y))
        return mses


class Polynom:
    """Represents a polynom of a specific dimension."""

    def __init__(self, dimension, X: np.ndarray, y: np.ndarray):
        """Initialize the class by computing the polynomials weights."""
        self.d = dimension
        self.w = self.fit(X=X, y=y)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Implement magic method so this class can be called right away."""
        return self.predict(X)

    def create_polymat(self, X: np.ndarray) -> np.ndarray:
        """Compute the polynomial matrix of size (#samples x dimension+1)."""
        X = self.dimension_correction(X)
        return X ** np.array([np.linspace(0, self.d, self.d + 1)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the vectors of weights of the polynomial (has size dimension x 1)."""
        y = self.dimension_correction(y)
        polymat = self.create_polymat(X=X)
        return np.linalg.pinv(polymat) @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output by the given data"""
        polymat = self.create_polymat(X=X)
        return polymat @ self.w

    def mse(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute mean squared error for given data."""
        y = self.dimension_correction(y)
        y_pred = self.predict(X)
        return np.mean(np.square(y_pred - y))

    @staticmethod
    def dimension_correction(v: np.ndarray) -> np.ndarray:
        """Make sure given vector (v) has at least 2D"""
        if len(v.shape) == 1:
            return v[:, np.newaxis]
        return v
