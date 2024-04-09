import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_angle(angle):
    """
    Normalizes an angle to the range [0, 2pi].
    :param angle: the angle to be normalized (in radians).
    :return: the normalized angle.
    """
    return angle - (angle + np.pi) % (2 * np.pi)


def eular_angles_to_rotation_matrix(angles):
    """
    Calculates the rotation matrix corresponding to the given Euler angles.
    :param angles: an array-like object of shape (3,) representing rotation angles around the x, y and z axes (in radians).
    :return: The rotation matrix corresponding to the given Euler angles.
    """
    return R.from_euler('xyz', angles).as_matrix()
