"""
This module contains function to run a singular value decomposition (SVD).
"""

__all__ = [
    "decompose",
]

from typing import Tuple
import numpy as np


def decompose(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the singular value decompositon to the given matrix.
    This process is done in 6 steps. Assume given matrix is A.

    1) Compute A^T * A
    2) Find eigenvalues and eigenvectors (with length 1) of computed matrix.
    3) Permute eigenvectors to order eigenvalues by size.
    4)

    :param matrix: Matrix to decompose.
    :return: the decomposed matrix A in 3 separate numpy.ndarray's. A = U*E*(V^T)
    """
    float_matrix = matrix.astype(float)
    min_dimension = min(float_matrix.shape)

    dot_product = np.transpose(float_matrix)@float_matrix
    eigenvalues, eigenvectors = np.linalg.eig(a=dot_product)
    sorting_index = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorting_index]

    V = eigenvectors[:, sorting_index]
    E = np.zeros_like(float_matrix)
    sqrt_diagonal_eigenvalues = np.identity(n=min_dimension) * np.sqrt(eigenvalues[:min_dimension])
    E[:min_dimension, :min_dimension] += sqrt_diagonal_eigenvalues
    U = orthonormalize(matrix=float_matrix@V)

    return U, E, np.transpose(V)


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """
    Applies the Gram-Schmidt process to a given matrix.
    If the given matrix is of shape (n x m) the resulting matrix will have shape (n x n).
    All columns of the given matrix will be normalized.
    :param matrix: Matrix to orthonormalize.
    :return: Matrix that consists of columns that are linearly independent to each other. Also each has length 1.
    """
    n, m = matrix.shape
    norm_matrix = normalize(matrix)
    if n > m:
        num_missing_dimensions = n - m
        for _ in range(num_missing_dimensions):
            new_basis = np.random.random(size=(n,))
            for basis in norm_matrix.transpose():
                new_basis = projection(vector=new_basis, basis=basis)
            norm_matrix = np.c_[norm_matrix, new_basis]
    elif n < m:
        norm_matrix = norm_matrix[:, :n]
    return norm_matrix


def projection(vector: np.array, basis: np.array) -> np.array:
    """
    Compute the projection of a given vector onto the given basis.
    :param vector: Vector to project on basis.
    :param basis: Basis to project the vector on.
    :return: Projected vector.
    """
    new_basis = vector - ((np.dot(basis, vector) / np.dot(basis, basis)) * basis)
    return normalize(matrix=new_basis)


def normalize(matrix: np.ndarray, axis=0) -> np.ndarray:
    """
    Normalize each column (axis=0) or row (axis=1) of the given matrix.
    This is done by summing the squares of each element along the axis, taking the square root of it and
    dividing each value by this value. The resulting length of the column/row will be 1.
    :param matrix: Matrix to normalize.
    :param axis: Axis to normalize along.
    :return: Normalized matrix.
    """
    return matrix / np.sqrt(np.sum(np.power(matrix, 2), axis=axis))
