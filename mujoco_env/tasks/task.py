from abc import ABC, abstractmethod

from gymnasium.core import ActType

from ..common.defs.types import InfoDict
from ..simulation.simulator import Simulator


class Task(ABC):
    def __init__(self, sim: Simulator):
        self.sim = sim

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """
        Reset the task parameters.
        :return: a dictionary of task information.
        """

    @abstractmethod
    def begin_frame(self, action: ActType) -> None:
        """
        A callback method called at the beginning of each simulation frame.
        :param action: The action to be taken in the coming step.
        """

    @abstractmethod
    def end_frame(self, action: ActType) -> None:
        """
        A callback method called at the end of each simulation frame.
        :param action: The action taken in the last step.
        """

    @abstractmethod
    def score(self) -> float:
        """
        Calculate the task-specific reward for the current state of the simulation.
        :return: the task-specific reward.
        """

    @abstractmethod
    def is_done(self) -> bool:
        """
        Check if the current state of the simulation is a terminal state.
        :return: `True` if the task is done, `False` otherwise.
        """

    @abstractmethod
    def get_info(self) -> InfoDict:
        """
        Get the task-specific information.
        :return: a dictionary of task information.
        """

    @abstractmethod
    def update_render(self, viewer) -> None:
        """
        Update the rendering with visual decoration task markers.
        :param viewer: a MuJoCo rendering object with "add_marker" method
        """
