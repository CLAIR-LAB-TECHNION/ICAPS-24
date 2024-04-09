import numpy as np

from .math import normalize_angle
from .misc import arraylike_func
from .defs.types import Pos3D


@arraylike_func
def position_euclidean_distance(x1: Pos3D, x2: Pos3D):
    """
    Calculates the Euclidean distance between two 3D positions.
    :param x1: an array-like object of shape (3,) representing the x, y and z coordinates of a 3D position.
    :param x2: same as `x1`.
    :return: ||x1 - x2||
    """
    return np.linalg.norm(x1 - x2)


@arraylike_func
def orientation_euclidean_distance(o1, o2):
    """
    Calculates the Euclidean distance between two 3D orientations. Orientation angles are normalized to be in range
    [0, 2pi], and the minimal distance between the angles is used.
    :param o1: an array-like object of shape (3,) representing rotation angles around the x, y and z axes (in radians).
    :param o2:an array-like object of shape (3,) representing rotation angles around the x, y and z axes (in radians).
    :return: same as `o1`
    """
    # normalize orientation angles to be in range [0, 2pi]
    o1_norm = normalize_angle(o1)
    o2_norm = normalize_angle(o2)

    # calculate the difference between the corresponding axis angles
    angle_diff = np.abs(o1_norm - o2_norm)

    # use complement angle if the difference is greater than pi
    agnle_diff_min = np.min([angle_diff, 2 * np.pi - angle_diff], axis=0)

    # return the norm of the angle difference
    return np.linalg.norm(agnle_diff_min)


@arraylike_func
def pose_euclidean_distance(p1, p2):
    """
    Calculates the Euclidean distance between two 3D poses (position and orientation).
    :param p1: an array-like object of shape (6,) of the form (x1, o1) where x1 = (x, y, z) represents the x, y and z
               coordinates of a 3D position and o1 = (rx, ry, rz) represents the rotation angles around the x,y and z
               axes (in radians).
    :param p2: same as `p1`.
    :return:
    """
    pos_dist = position_euclidean_distance(p1[:3], p2[:3])
    rot_dist = orientation_euclidean_distance(p1[3:], p2[3:])

    return pos_dist + rot_dist
