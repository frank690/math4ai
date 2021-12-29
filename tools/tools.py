"""
This module contains a variety of functions I like to use from time to time.
"""

__all__ = [
    "numerical_gradient",
    "draw_learning_curve",
]


from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure


def numerical_gradient(
    f: Callable, key: str, eps: float = 1e-10, **kwargs
) -> np.ndarray:
    """
    compute the gradient of the given function numerically.
    :param f: function whose gradient we want to compute.
    :param key: Name of key variable (in f) to alter with eps to compute gradient for gradual change.
    :param eps: incremental step to compute gradient.
    :param kwargs: arguments that will be passed to f.
    :Note: It is assumed that the given key variable in kwargs is a numpy.ndarray.
    This matrix will be converted to np.float64.
    :return: computed gradient of given function f with respect to key variable.

    Example:
    def some_function(w, X, b):
        return (w.T@X + b)/np.sqrt(w.T@w)

    X = np.array([[0, 1, 2, 3, 3], [2, 2, 2, 2, 5]])
    w = np.array([[1], [-1]])
    b = 1

    numerical_gradient(f=some_function, key='w', w=w, X=X, b=b)
    """
    base_result = f(**kwargs)
    kwargs[key] = kwargs[key].astype(np.float64)
    flat_key_values = kwargs[key].flatten()
    incremental_results = []
    for idx, val in enumerate(flat_key_values):
        key_values = flat_key_values.copy()
        key_values[idx] += eps
        key_values = np.reshape(key_values, kwargs[key].shape)
        incremental_results.append(f(**{**kwargs, key: key_values}))

    incremental_results = np.stack(incremental_results, axis=1)
    return np.squeeze((incremental_results - base_result) / eps)


def draw_learning_curve(
    data: np.ndarray, axis: Optional[Axes] = None
) -> Optional[Figure]:
    """
    Draws the learning curve of the given data array.
    :param data: Cost trend to plot.
    :param axis: Axis of figure to update with the given data.
    :return: Newly created figure containing the learning curve iff no axis was given.
    """
    if axis is None:
        figure = plt.figure()
        axis = figure.add_subplot(111)
    else:
        figure = None

    axis.set_title("Learning Curve", fontsize=18)
    axis.set_xlabel("Number of iterations", fontsize=15)
    axis.set_ylabel("Loss", fontsize=15)
    axis.plot([i for i in range(len(data))], data)

    return figure
