from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .base_spec import Spec
from ...common.defs.types import ActuatorType, JointType
from ...common.defs.types import ParamsDict
from ...common.misc import set_iterable_arg


# TODO consolidate elements to ElementSpec with `tag`, `attrs`, and `children` fields, and attach function.


@dataclass(frozen=True, eq=False)
class JointSpec(Spec):
    """
    A specification for a joint MJCF element.

    public fields:
        - type: The type of joint (see spear_env.common.defs.types.JointType).
        - attrs: A dictionary of attributes for the joint element, as in MJCF.
        - actuators: A collection of actuators to be linked to the joint.
    """
    type: JointType = JointType.FREE
    attrs: ParamsDict = field(default_factory=dict)
    actuators: Sequence[ActuatorSpec] = tuple()

    def __post_init__(self):
        # Cast input type to JointType
        if not isinstance(self.type, JointType):
            super().__setattr__('type', JointType(self.type))

        if 'name' not in self.attrs:
            self.attrs['name'] = f'user_joint_{self.id}'

        # set the actuators to a tuple of ActuatorSpecs
        super().__setattr__('actuators', set_iterable_arg(ActuatorSpec, self.actuators))


@dataclass(frozen=True, eq=False)
class ActuatorSpec(Spec):
    """
    A specification for an actuator MJCF element.

    public fields:
        - type: The type of actuator (see spear_env.common.defs.types.ActuatorType).
        - attrs: A dictionary of attributes for the actuator element, as in MJCF.
    """
    type: ActuatorType
    attrs: ParamsDict = field(default_factory=dict)

    def __post_init__(self):
        # Cast input type to JointType
        if not isinstance(self.type, ActuatorType):
            super().__setattr__('type', ActuatorType(self.type))

        if 'name' not in self.attrs:
            self.attrs['name'] = f'user_actuator_{self.id}'
