from __future__ import annotations

from typing import Union, TypeVar

from .specs import *
from ..common.defs.cfg_keys import *
from ..common.defs.types import Asset, Config

# represents the name of an internal asset or a configuration for such an asset
AssetOrConfig = Union[Asset, Config]

# the generic type that inherits from Spec
SpecT = TypeVar('SpecT', bound=Spec)


def episode_from_cfg(cfg: Config) -> EpisodeSpec:
    """
    Converts a configuration dictionary to an episode specification object.

    A configuration dictionary defines the entire episode using different sub-specifications. Dictionary keys are
    strings and values match those of the specification dataclasses from the `spear_env.episode.specs` sub-package. A
    nested specification may be defined using a configuration dictionary with the keys corresponding to that
    specification's dataclass. All such keys are provided as constants in `spear_env.defs.cfg_keys`. Specifications that
    can be initialized wiith a single parameter may be defined by that parameter in the configuration.

    >>> from mujoco_env.episode import RobotSpec
    >>> episode_from_cfg({
    ...     'scene': 'floatworld',
    ...     'robot': RobotSpec('ur5e'),
    ...     'task': {
    ...         'cls': 'spear_env.tasks:NullTask'
    ...         # missing 'params' is set as default `{}` as in `spear_env.episode.specs.task_spec.TaskSpec`
    ...     }
    ... })  # doctest:+ELLIPSIS
    EpisodeSpec(id=..., scene=SceneSpec(...), robot=RobotSpec(...), task=TaskSpec(...)

    :param cfg: the configuration dictionary to convert.
    :return: an episode specification matching the given values in `cfg` and spec defaults.
    """
    scene = scene_spec_from_name_or_cfg(cfg[SCENE])
    robot = robot_spec_from_name_or_cfg(cfg[ROBOT])
    task = task_spec_from_name_or_cfg(cfg[TASK])

    return EpisodeSpec(scene, robot, task)


def scene_spec_from_name_or_cfg(asset_or_cfg: AssetOrConfig) -> SceneSpec:
    """
    Converts a configuration dictionary or resource name to a scene specification object.
    See `spear_env.episode.cfg_parsing.episode_from_cfg` for more details.
    :param asset_or_cfg: the scene asset name or a scene configuration dictionary
    :return:a scene specification object.
    """
    if isinstance(asset_or_cfg, dict) and SCENE_OBJECTS in asset_or_cfg:
        asset_or_cfg[SCENE_OBJECTS] = __list_or_single_from_name_or_cfg(asset_or_cfg[SCENE_OBJECTS], ObjectSpec)

    return __type_from_name_or_cfg(asset_or_cfg, SceneSpec)


def robot_spec_from_name_or_cfg(asset_or_cfg: AssetOrConfig) -> RobotSpec:
    """
    Converts a configuration dictionary or resource name to a robot specification object.
    See `spear_env.episode.cfg_parsing.episode_from_cfg` for more details.
    :param asset_or_cfg: the robot asset name or a scene configuration dictionary
    :return:a robot specification object.
    """
    # if received config dict
    if isinstance(asset_or_cfg, dict) and ROBOT_ATTACHMENTS in asset_or_cfg:
        asset_or_cfg[ROBOT_ATTACHMENTS] = __list_or_single_from_name_or_cfg(asset_or_cfg[ROBOT_ATTACHMENTS],
                                                                            AttachmentSpec)

    return __type_from_name_or_cfg(asset_or_cfg, RobotSpec)


def task_spec_from_name_or_cfg(asset_or_cfg: AssetOrConfig) -> TaskSpec:
    """
    Converts a configuration dictionary or an entrypoint string to a task specification object.
    See `spear_env.episode.cfg_parsing.episode_from_cfg` for more details.
    :param asset_or_cfg: the task entrypoint or a task configuration dictionary
    :return:a task specification object.
    """
    return __type_from_name_or_cfg(asset_or_cfg, TaskSpec)


def __list_or_single_from_name_or_cfg(asset_or_cfg: Union[AssetOrConfig, list[AssetOrConfig]],
                                      spec_type: type[SpecT]) -> list[SpecT]:
    if isinstance(asset_or_cfg, list):
        return [__type_from_name_or_cfg(data, spec_type) for data in asset_or_cfg]
    else:
        return [__type_from_name_or_cfg(asset_or_cfg, spec_type)]


def __type_from_name_or_cfg(inp: AssetOrConfig | SpecT, spec_cls: type[SpecT]) -> SpecT:
    if isinstance(inp, spec_cls):
        return inp  # correct spec type. no extra input required.
    if isinstance(inp, dict):
        if ADDON_BASE_JOINTS in inp:
            inp[ADDON_BASE_JOINTS] = __list_or_single_from_name_or_cfg(inp[ADDON_BASE_JOINTS], JointSpec)
        if JOINT_ACTUATORS in inp:
            inp[JOINT_ACTUATORS] = __list_or_single_from_name_or_cfg(inp[JOINT_ACTUATORS], ActuatorSpec)
        return spec_cls(**inp)
    else:
        return spec_cls(inp)
