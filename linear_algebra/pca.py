"""
This module contains function to run a principal component analysis (PCA).
"""

__all__ = [
    "pca",
]

import numpy as np


def pca(matrix: np.ndarray, dimension: int) -> np.ndarray:
    """
    Run the principal component analysis and return a matrix that reduces any given datapoint from the vector space of
    the given matrix to a vector space of the desired dimension.
    A PCA is computed in several steps.
    1) Compute the mean of the given matrix (X).
    2) Center the data by subtracting the mean (X - mean(X)).
    3) Compute covariance matrix of X*X^T.
    4) Compute eigenvalues of covariance matrix.
    5) Use n biggest eigenvalues to compute eigenvectors (n = target dimension).
    6) return the product of the matrix times the matrix of eigenvectors.
    Note (!): It is assumed that your features are represented in the columns of the given matrix. That means that
    (given you have more samples than features) the number of rows should be greater than the number of columns.
    If this is not the case, please transpose your matrix before passing it into this function.
    :param matrix: Matrix to derive PCA from.
    :param dimension: Target dimension that the PCA should reduce the input space to.
    :return: A matrix that can be used to reduce the dimension of samples from the original (unreduced) vector space.
    """
    n, m = matrix.shape
    feature_means = np.mean(matrix, axis=0)
    centered_matrix = matrix - feature_means
    covariance_matrix = (1 / (n - 1)) * np.transpose(centered_matrix) @ centered_matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_eigenvalues_index = np.argsort(eigenvalues)[::-1]
    desired_eigenvalues_index = sorted_eigenvalues_index[:dimension]
    return eigenvectors[:, desired_eigenvalues_index]
