from __future__ import annotations

from dataclasses import dataclass

from .base_spec import Spec
from .robot_spec import RobotSpec
from .scene_spec import SceneSpec
from .task_spec import TaskSpec


@dataclass(frozen=True, eq=False)
class EpisodeSpec(Spec):
    """
    A specification for an episode to be run in the environment. It defines the simulated scene, the robot agent, and
    the task to be performed.

    public fields:
        - scene: A specification of the scene to be simulated
        - robot: A specification of the robot agent in the simulation.
        - task: A specification of the task to be performed.
    """
    scene: SceneSpec
    robot: RobotSpec
    task: TaskSpec

    def __post_init__(self):
        if not isinstance(self.scene, SceneSpec):
            super().__setattr__('scene', SceneSpec(self.scene))
        if not isinstance(self.robot, RobotSpec):
            super().__setattr__('robot', RobotSpec(self.robot))
        if not isinstance(self.task, TaskSpec):
            super().__setattr__('task', TaskSpec(self.task))

    @staticmethod
    def require_different_models(episode1: 'EpisodeSpec', episode2: 'EpisodeSpec') -> bool:
        return (SceneSpec.require_different_models(episode1.scene, episode2.scene) or
                RobotSpec.require_different_models(episode1.robot, episode2.robot))
