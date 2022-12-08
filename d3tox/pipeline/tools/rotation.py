import numpy as np


def rotate_2d_points_over_center(points_xy, center_xy, angle):
    """
    Rotate points over a center in cartesian space.

    points_xy:
        The points that are going to be rotated

    center_xy:
        The point to rotate around

    angle:
        The angle in radians
    """

    # center on center_xy in cartesian space
    points_xy = points_xy - center_xy

    # counter-clockwise rotation using rotation matrix
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])

    # vectorized rotation
    # @ is the new dot-product notation
    # transformation is arranged in standard form
    rv = (r @ points_xy.T).T

    # return rv, the result of the rotation
    rv = rv + center_xy
    return rv
