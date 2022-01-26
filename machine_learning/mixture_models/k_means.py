"""This module holds classes and functions to run the k-means algorithm."""

__all__ = [
    "k_means",
]

from typing import Optional

import numpy as np


def k_means(data: np.ndarray, k: int = 3, iterations: int = 5):
    """
    Implementation of the naive k-means algorithm.
    Try to iterativly find k clusters in the given data until the change is smaller or equal than eps.
    :param data: Data to apply kmeans algorithm to.
    :param k: Number of desired clusters.
    :param iterations: Number of times to run the algorithm.
    :Note: It is assumed that each row in the given data represents a sample.
    """
    means = None
    grouping = None

    for iteration in range(iterations):
        means = _compute_cluster_means(data=data, k=k, grouping=grouping)
        distances = _compute_distances(data=data, k=k, means=means)
        grouping = np.argmin(distances, axis=1)
    return means, grouping


def _distance_to_point(data: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Compute the squared distance of each sample (row) n in the given data to the given point.
    :param data: Samples to compute distance to point to.
    :param point: Point to compute distance to.
    :return: 1D Numpy array of length n that holds all distances to the given point.
    """
    return np.array([np.sum(np.square(sample - point)) for sample in data])


def _compute_cluster_means(
    data: np.ndarray, k: int, grouping: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the mean values of each cluster (group) or initialize the mean by assigning a random (unique) data point.
    :param data: Data to compute means from. Each row should be a sample.
    :param k: Number of clusters (means) to compute.
    :param grouping: Numpy array indicating what sample belong to what group.
    :return: Mean of each cluster due to their assigned samples.
    """
    _, d = data.shape

    if grouping is None:
        rng = np.random.default_rng()
        means = rng.choice(np.unique(data, axis=0), k, replace=False)
    else:
        means = np.zeros((k, d))
        for dim in range(k):
            means[dim, :] = np.mean(data[grouping == dim, :], axis=0)
    return means


def _compute_distances(data: np.ndarray, k: int, means: np.ndarray) -> np.ndarray:
    """
    Compute the (squared) distance of each sample (of the given data) to the mean of each cluster.
    :param data: Data to compute means from. Each row should be a sample.
    :param k: Number of clusters (means) to compute.
    :param means: Position of the means of each cluster.
    :return: The distance of each sample to each cluster.
    """
    n, _ = data.shape
    distances = np.zeros((n, k))

    for dim in range(k):
        distances[:, dim] = _distance_to_point(data=data, point=means[dim, :])
    return distances


if __name__ == "__main__":
    """run some demo of the k-means algorithm by downsizing a given picture."""
    import matplotlib.image as img

    k = 5
    iterations = 10
    image = img.imread("../../data/road.png")[:, :, :3]  # ignore alpha
    w, h, d = image.shape
    image = image.reshape((w * h, d))
    means, grouping = k_means(data=image, k=k, iterations=iterations)
    most_frequent_color_by_group = np.array(
        [np.median(image[grouping == group], axis=0) for group in range(k)]
    )
    unrolled_reduced_image = np.array(
        [most_frequent_color_by_group[group] for group in grouping]
    )
    reduced_image = unrolled_reduced_image.reshape((w, h, d))
    img.imsave("reduced_road.png", reduced_image)
