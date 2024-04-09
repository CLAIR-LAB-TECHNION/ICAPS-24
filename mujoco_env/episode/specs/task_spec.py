from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from .base_spec import Spec
from ...common.defs.types import ParamsDict
from ...common.misc import load_from_entrypoint
from ...tasks.task import Task


@dataclass(frozen=True, eq=False)
class TaskSpec(Spec):
    """
    A specification for a task to perform in the environment simulation.

    public fields:
        - cls: The class of the task to be performed. This can be a string, in which case it is interpreted as an
               entrypoint from which to load a task, e.g., `my_package.my_module:MyTaskClass`.
        - params: A dictionary of parameters to be passed to the task upon reset.
    """
    cls: Type[Task]
    params: ParamsDict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.cls, str):
            super().__setattr__('cls', load_from_entrypoint(self.cls))
