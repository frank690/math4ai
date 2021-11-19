"""This module contains functions regarding the hull of a convex function."""

import numpy as np
from typing import List, Tuple


def min_list(data: List[Tuple]) -> List[Tuple]:
    """
    returns the minimum list of 2D datapoints that span a convex hull.
    :param data: datapoints of convex set.
    :return: minimum datapoints that span the convex hull.
    """
    hull = []
    poi = min(data, key=lambda x: x[1])
    angle = -1

    while poi not in hull:
        hull.append(poi)
        modified_datapoints = substract_coordinates(points=datapoints, point=poi)
        next_poi, angle = smallest_counter_clockwise_angle(points=modified_datapoints, previous_angle=angle)
        poi = tuple(map(sum, zip(next_poi, poi)))

    return hull


def smallest_counter_clockwise_angle(points: List[Tuple], previous_angle: float) -> Tuple:
    """
    computes the ccw angle from the center to each point in the list of points.
    returns the point that has the smallest angle (and is the furthest away).
    :param points: list of datapoints.
    :param previous_angle: angle from previous datapoint to current. use as offset to find next smallest angle.
    :return: datapoint with the smallest angle
    (and furthest distance in case of identical angle) and said smallest angle.
    """
    angles_and_distances = [compute_angle_and_distance(point) for point in points]

    smallest_angle = 360
    biggest_distance = 0
    next_point_idx = 0
    for idx, (angle, distance) in enumerate(angles_and_distances):
        if angle > previous_angle:
            if (angle < smallest_angle) or ((angle == smallest_angle) & (distance > biggest_distance)):
                next_point_idx = idx
                smallest_angle = angle
                biggest_distance = distance
    return points[next_point_idx], smallest_angle


def compute_angle_and_distance(point: Tuple) -> Tuple[float, float]:
    """
    compute the angle and distance of the given point.
    :param point: 2d datapoint with x and y coordinate.
    :return: angle and distance of the given point.
    """
    x, y = point
    if (x == 0) and (y == 0):
        return 360, 0
    return np.rad2deg(np.arctan2(y, x)) % 360, np.sqrt(x**2 + y**2)


def substract_coordinates(points: List[Tuple], point: Tuple) -> List[Tuple]:
    """
    substract the coordinates of a point from each point in the list of datapoints.
    :param points: list of datapoints.
    :param point: reference datapoint.
    :return: modified list of datapoints.
    """
    x0, y0 = point
    return [(x - x0, y - y0) for x, y in points]
