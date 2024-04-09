from abc import ABC
from dataclasses import dataclass
from typing import Optional

from gymnasium.core import ActType

from ..task import Task
from ...common import Pos3D, Rot3D, Vector


@dataclass
class ObjectConfiguration:
    position: Pos3D
    rotation: Rot3D = None
    joint_positions: Vector = None
    joint_velocities: Vector = None


class RearrangementTask(Task, ABC):
    def reset(self,
              target_objects: dict[str, ObjectConfiguration],
              other_objects: Optional[dict[str, ObjectConfiguration]] = None,
              tolerance: float = 0.1) -> None:
        pass

    def begin_frame(self, action: ActType) -> None:
        pass

    def end_frame(self, action: ActType) -> None:
        pass
