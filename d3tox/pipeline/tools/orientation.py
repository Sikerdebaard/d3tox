import numpy as np
import logging

from scipy.optimize import curve_fit, minimize
from d3tox.pipeline.tools.rotation import rotate_2d_points_over_center


def estimate_img_orientation(centroids, x_idx, y_idx):
    """
    Estimate the image orientation based on a desired outcome of having C2 on the left of the image and C7 on the right.
    It also estimates the mid-point over multiple centroids to rotate over.
    """

    points_xy = centroids.loc[[x_idx, y_idx]].T.values

    # curve fit func _linear_line on centroids x and centroids y
    popt, _ = curve_fit(_linear_line, points_xy[:, 0], points_xy[:, 1])

    x_mid = np.mean([centroids.loc[x_idx].min(), centroids.loc[x_idx].max()])
    y_mid = _linear_line(x_mid, popt[0], popt[1])

    center_xy = np.array((x_mid, y_mid))

    angles = _optim1(points_xy, center_xy), _optim2(points_xy, center_xy)

    if _find_angle_between_radians(angles, absolute=True) > 5:
        # if the angle-difference between the two > 5 then _optim2 has failed
        # so we pick the slightly less optimized result from _optim1
        logging.info('_optim1 / _optim2 angle > 5 degrees apart, falling back to _optim1 result')
        angle = angles[0]
    else:
        # pick the best out of the _optim1 and _optim2 results
        angle = _pick_lowest_score(angles, points_xy, center_xy)

    return angle, center_xy


def _find_angle_between_radians(radians, absolute=True):
    """
    Returns the distance between two radians in degrees of separation
    """
    angles = np.degrees(radians)

    a = angles[0] - angles[1]
    a = (a + 180) % 360 - 180

    if absolute:
        a = abs(a)

    return a


def _pick_lowest_score(angles, points_xy, center_xy):
    scores = []
    for i in range(len(angles)):
        scores.append(_orientation([np.degrees(angles[i])], points_xy, center_xy))

    return angles[scores.index(min(scores))]


def _optim1(points_xy, center_xy):
    scores = []
    # iterate over all angles 0 to 359 and store the scores
    for i in range(0, 360):
        scores.append(_orientation([i], points_xy, center_xy))

    angle = np.radians(scores.index(min(scores)))

    return angle


def _optim2(points_xy, center_xy):
    res = minimize(_orientation, np.array([900.]), (points_xy, center_xy), method='Nelder-Mead')  # , bounds=[[0., 360]])
    angle = np.radians(res.x % 360)[0]

    return angle


def _orientation(angle, points_xy, center_xy):
    angle = np.radians(angle[0])
    points = rotate_2d_points_over_center(points_xy, center_xy, angle)

    # curve fit funct _linear_line on centroids x and centroids y
    popt, _ = curve_fit(_linear_line, points[:, 0], points[:, 1])

    # try to minimize the line slope and make sure the C2 (or closest) vertebrae is on the left side of the image
    score = abs(popt[0]) * 100 + int(points[0, 0] < points[1, 0]) + int(points[0, 0] < points[2, 0])
    return score


def _linear_line(x, a, b):
    return a * x + b
