"""This module holds all functions and classes dealing with the gaussian distribution."""


__all__ = [
    "pdf",
]


import numpy as np


def pdf(X: np.ndarray, m: np.ndarray, C: np.ndarray) -> float:
    """
    Compute the probabilities that the given points belong to the multivariate gaussian distribution
    defined by the given parameters m (mean) and C (covariance).
    :param X: data points (shape n x d). Each row represents a sample.
    :param m: Mean of gaussian (shape 1 x d).
    :param C: Covariance matrix of gaussian (shape d x d).
    :return: Probabilities that the given points (X) belong to this distribution.
    """
    return np.atleast_2d(
        np.diag(
            (1 / np.sqrt(np.power(2 * np.pi, X.shape[1]) * np.linalg.det(C)))
            * np.exp(-0.5 * (X - m.T) @ np.linalg.inv(C) @ (X - m.T).T)
        )
    )
