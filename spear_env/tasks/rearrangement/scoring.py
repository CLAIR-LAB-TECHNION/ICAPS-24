import numpy as np
from numpy.typing import ArrayLike

from ...common.metrics import position_euclidean_distance


def position_epsilon_success_score(x1: ArrayLike, x2: ArrayLike, ep: float) -> int:
    """
    Determines success or failure of a position-based task by comparing the euclidean distance between two positions.
    :param x1: an array-like object of shape (3,) representing the x, y and z coordinates of a 3D position.
    :param x2: same as `x1`.
    :param ep: a scalar threshold distance between the two positions for the task to be considered a success
    :return: 1 if ||x1 - x2|| < eps, 0 otherwise.
    """
    return int(position_euclidean_distance(x1, x2) < ep)


def multi_object_position_epsilon_success_score(x1: ArrayLike, x2: ArrayLike, epsilon: float) -> int:
    """
    Determines success or failure of a position-based task by comparing the euclidean distance between two positions
    for multiple objects.
    :param x1: an array-like object of shape (N, 3) representing the x, y and z coordinates of 3D positions of N
               objects.
    :param x2: same as `x1`.
    :param epsilon: a scalar threshold distance between the two positions for the task to be considered a success
    :return: 1 if ||x1 - x2|| < eps, 0 otherwise.
    """
    return int(np.all(position_euclidean_distance(x1, x2) < epsilon))


def discounted_return(rewards: ArrayLike, gamma: float, normalize: bool = False) -> float:
    """
    Calculates the discounted return for a sequence of rewards.
    :param rewards: an array-like object of shape (N,) representing the sequence of rewards.
    :param gamma: the discount factor.
    :param normalize: whether to normalize the discounted return.
    :return: the discounted return.
    """
    discounted_return = 0
    for i, reward in enumerate(rewards):
        discounted_return += gamma ** i * reward
    if normalize:
        discounted_return /= np.sum(gamma ** np.arange(len(rewards)))

    return discounted_return
