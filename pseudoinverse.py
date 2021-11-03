import numpy as np


def pre_checks(matrix: np.ndarray):
    """
    Apply multiple checks to see if matrix is invertible.
    :param matrix: Matrix to check.
    """
    assert isinstance(matrix, np.ndarray), print("Given matrix is not of expected type numpy.ndarray.")
    rows, columns = matrix.shape
    assert rows == columns, print("The matrix you provided is not square and can thus not be an inverted.")
    assert np.linalg.det(matrix) != 0, print("Matrix determinant is 0. Thus the matrix is not invertible.")


def fix_diagonal(matrix: np.ndarray, inverse: np.ndarray):
    """
    Checks if the values on the main diagonal of the given matrix are non-zero.
    In case they are, the first row that is non-zero at this column is added.
    :param matrix: Matrix to prepare.
    :param inverse: Original identity matrix that is also changed accordingly.
    This will later result in the final inverse.
    """
    diagonal = np.diagonal(matrix)
    for row_index, diagonal_value in enumerate(diagonal):
        if diagonal_value == 0:
            index_non_zero_row = np.nonzero(matrix[:, row_index])[0][0]
            matrix[row_index] += matrix[index_non_zero_row]
            inverse[row_index] += inverse[index_non_zero_row]


def eliminate_non_diagonals(matrix: np.ndarray, dimension: int, inverse: np.ndarray):
    """
    Apply row-operations on the given matrix to reduce all non-diagonal values to zero.
    Keep track of what was done by applying the same operations to the given inverse matrix.
    :param matrix: Matrix to delete non-diagonals from.
    :param dimension: Dimension of the matrix.
    :param inverse: Inverse matrix to keep track of the operations.
    """
    for column_index in range(dimension):
        for row_index in range(dimension):
            if row_index != column_index:
                inverse[row_index] -= \
                    matrix[row_index][column_index] * inverse[column_index] / matrix[column_index][column_index]
                matrix[row_index] -= \
                    matrix[row_index][column_index] * matrix[column_index] / matrix[column_index][column_index]


def divide_by_diagonal(matrix: np.ndarray, inverse: np.ndarray):
    """
    Divides each value of a row by its corresponding diagonal value.
    :param matrix: Matrix that holds diagonal and needs to be divided.
    :param inverse: The nearly-done inverse to keep track of changes on the given matrix.
    """
    diagonal = np.diagonal(matrix)
    for row_num, diagonal_value in enumerate(diagonal):
        matrix[row_num] /= diagonal_value
        inverse[row_num] /= diagonal_value


def invert(matrix: np.ndarray) -> np.ndarray:
    """
    Finds inverse of given matrix.
    :param matrix: Matrix to find inverse of.
    :return: The inverse of the given matrix.
    """
    dimension = matrix.shape[0]
    matrix_to_invert = matrix.copy()
    inverse = np.eye(dimension)

    pre_checks(matrix=matrix_to_invert)
    fix_diagonal(matrix=matrix_to_invert, inverse=inverse)
    eliminate_non_diagonals(matrix=matrix_to_invert, dimension=dimension, inverse=inverse)
    divide_by_diagonal(matrix=matrix_to_invert, inverse=inverse)
    return inverse


def matrix_distance(mat_A, mat_B):
    sum_of_differences = 0
    for i in range(len(mat_B)):
        for j in range(len(mat_B[i])):
            sum_of_differences += abs(mat_A[i, j] - mat_B[i, j])
    print(f"Sum of differences: {sum_of_differences}")
    return sum_of_differences


if __name__ == '__main__':
    A = np.array([[1., 2., 3.], [1., 1., 1.], [5., 2., 1.]])
    inverse_A = invert(matrix=A)
    B = np.eye(3)

    for C in [inverse_A]:
        B = B@C

    assert matrix_distance(mat_A=A, mat_B=B) < 0.01, print("Inversion check failed.")

