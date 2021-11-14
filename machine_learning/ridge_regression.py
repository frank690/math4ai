import numpy as np
from random import Random
from typing import Tuple, List


def ridgeCV(X: np.ndarray, y: np.ndarray, number_of_folds: int, lambda_grid: List[int]) -> Tuple[np.ndarray, float, float]:
    best_mse = best_lamb = np.inf

    for lamb in lambda_grid:
        cv_mse = []
        cv_weights = []
        for X_training, y_training, X_validation, y_validation in kfold(X=X, y=y, folds=number_of_folds):
            rlr = RidgeLinearRegression(regularization=lamb)
            rlr.fit(X=X_training, y=y_training)
            y_pred = rlr.predict(X=X_validation)
            cv_mse.append(np.mean(np.square(y_pred - y_validation)))
            cv_weights.append(rlr.w)

        cv_mse = np.mean(cv_mse)
        if cv_mse < best_mse:
            best_mse = cv_mse
            best_lamb = lamb

    rlr = RidgeLinearRegression(regularization=best_lamb)
    rlr.fit(X=X, y=y)
    best_w = rlr.w

    return best_w, best_lamb, best_mse


def separate_features_from_target(data: np.ndarray, target_column: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separates the given data into features (input data) and the target (output data).
    By default it is assumed that the last column contains the target data.
    :param data: data to separate into features and target.
    :param target_column: Index of column that contains target data.
    :return: feature and target as numpy.ndarrays (in this order)
    """
    features = data[:, :target_column]
    target = data[:, target_column]
    return features, target[:, np.newaxis]


def kfold(X: np.ndarray, y: np.ndarray, folds: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fold the given data in k (random) folds.
    note that row-wise samples are expected.
    :param X: input data (features) to fold
    :param y: output data (target) to fold
    :param folds: number of folds to do.
    :return: four numpy.ndarrays that represent training data (X and y) and testing data (X and y) in this order.
    """
    num_samples = X.shape[0]
    indices = [index for index in range(num_samples)]
    Random(42).shuffle(indices)
    splits = np.array_split(indices, folds)
    for split in splits:
        mask = np.zeros(num_samples, dtype=bool)
        mask[split] = True
        yield X[~mask, :], y[~mask, :], X[mask, :], y[mask, :]


class RidgeLinearRegression:
    """
    Class to run ridge linear regression
    """
    def __init__(self, regularization: float = 0):
        self.w = None
        self.regularization = regularization

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Derive the weights from the given data."""
        self.w = np.linalg.pinv((np.eye(X.T.shape[0]) * self.regularization) + X.T @ X) @ X.T @ y

    def predict(self, X: np.ndarray):
        """Predict an output given some input."""
        return X @ self.w


if __name__ == '__main__':
    data = np.genfromtxt("../data/boston.txt", dtype=float, skip_header=22)
    testing_data = data[:100, :]
    training_data = data[100:, :]
    X, y = separate_features_from_target(training_data)
    w, best_lambda, best_score = ridgeCV(X=X, y=y, number_of_folds=5, lambda_grid=[0.001, 0.1, 1, 10, 100])
    print(f"Best lambda: {best_lambda} \nBest error: {best_score} \nBest weights: {w}")
