from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from os import PathLike
from typing import Annotated, Literal, Any, Union, Type

from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings

__all__ = [
    'FilePath',
    'Asset',
    'Entrypoint',
    'InfoDict',
    'ParamsDict',
    'Config',
    'Identifier',
    'Scalar',
    'Vector',
    'Pos3D',
    'Rot3D',
    'JointType',
    'ActuatorType',
    'AssetType'
]

# a path to a file. variables of this type will be treated as a valid (not necessarily existing) path to a file
FilePath = Union[str, PathLike[str]]

# a FilePath or the name of an existing internal asset
Asset = Union[str, FilePath]

# a string that describes the import path to a class or function.
# see `spear_env.common.misc.load_from_entrypoint` for more details.
Entrypoint = str

# Identical representations of different specification dictionaries.
# - InfoDict: contains additional information about an environment, task, etc.
# - ParamsDict: a collection of keyword arguments for a callable.
# - Config: a configuration dictionary for describing environment options.
InfoDict = ParamsDict = Config = dict[str, Any]

# a general identifier of an MJCF element in the form of a name or index.
Identifier = Union[str, int]

# A numerical value
Scalar = Union[float, int]

# A general purpose numeric vector type
Vector = Sequence[Scalar]

# A vector of shape (3,) representing a 3d position in space
Pos3D = Annotated[
    Sequence[float],
    Literal[3]
]

# A vector of shape (3,) representing a 3d rotation in space using euler angles
EulerRot3D = Annotated[
    Sequence[float],
    Literal[3]
]

# A vector of shape (4,) representing a 3d rotation in space as a unit quaternion
QuatRot3D = Annotated[
    Sequence[float],
    Literal[4]
]

# A vector of shape (6,) representing a 3d rotation in space by specifying the X and Y axes. (Z = cross(X,Y))
XYAxesRot3D = Annotated[
    Sequence[float],
    Literal[6]
]

# A vector that represents 3d rootation in space
Rot3D = Union[
    EulerRot3D,
    QuatRot3D,
    XYAxesRot3D
]


def __str_list_to_enum(enum_name, field_list: list[str]) -> Type[Enum]:
    values_map = {filed.upper(): filed.lower() for filed in field_list}
    return Enum(enum_name, values_map)


# An Enum of all joint types with string values.
# The actual values can be found in `dm_control.mujoco.wrapper.mjbindings.enums.mjtJoint._fields` omitting the "mjJNT"
# prefix from the field name.
JointType = __str_list_to_enum('JointType',
                               list(map(lambda s: s.replace('mjJNT_', ''), mjbindings.enums.mjtJoint._fields)))

# An Enum of possible actuator tags with string values
# The actual values can be found in `dm_control.mjcf.traversal_utils._ACTUATOR_TAGS`
ActuatorType = __str_list_to_enum('ActuatorType', list(mjcf.traversal_utils._ACTUATOR_TAGS))


class AssetType(Enum):
    """The different types of assets in the assets directory. This is used to determine the file path of an asset"""
    SCENE = 'scene'  # an MJCF file that defines a scene
    ROBOT = 'robot'  # an MJCF file that defines a robot
    ATTACHMENT = 'attachment'  # an MJCF file that defines a robot attachment
    MOUNT = 'mount'  # an MJCF file that defines a robot mount
    OBJECT = 'object'  # an MJCF file that defines a scene object
    EPISODE = 'episode'  # a YAML file that defines an episode (scene, robot, task)
