from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .addon_spec import AddonSpec, InitStateAddonSpec
from .camera_spec import CameraSpec
from ...common.defs.defaults import DEFAULT_ATTACHMENT_SITE_NAME
from ...common.defs.types import AssetType, Asset
from ...common.misc import set_iterable_arg


@dataclass(frozen=True, eq=False)
class RobotSpec(InitStateAddonSpec):
    """
    A specification for a robot agent.

    public fields:
        - attachments: A collection of specifications for attachments to the robot.
        - mount: A specification for an asset on which to mount the robot.
        - privileged_info: Whether the robot has access to privileged simulator information.

    static methods:
        - require_different_models: Check if two robots require instantiating different simulation models.
    """
    cameras: Sequence[CameraSpec] = tuple()
    attachments: Sequence[AttachmentSpec] = tuple()
    mount: Optional[MountSpec] = None
    privileged_info: bool = False

    # TODO specify sensors / cameras / actuators

    def __post_init__(self):
        super().__post_init__()

        # set cameras and attachments to a tuple of the corresponding spec types
        super().__setattr__('cameras', set_iterable_arg(CameraSpec, self.cameras))
        super().__setattr__('attachments', set_iterable_arg(AttachmentSpec, self.attachments))

        # Cast input mount to MountSpec

        if self.mount is not None and isinstance(self.mount, str):
            super().__setattr__('mount', MountSpec(self.mount))

    @property
    def _asset_type(self) -> AssetType:
        return AssetType.ROBOT

    @staticmethod
    def require_different_models(robot1: 'RobotSpec', robot2: 'RobotSpec') -> bool:
        """
        Check if two robots require instantiating different simulation models. That is, if we were to exchange one for
        the other, check if it would be necessary to load a new model into the simulation.
        :param robot1: a robot to compare.
        :param robot2: another robot to compare.
        :return: True if the robots require different models, False otherwise.
        """
        return (AddonSpec.require_different_models(robot1, robot2) or
                AddonSpec.collection_require_different_models(robot1.attachments, robot2.attachments) or
                AddonSpec.require_different_models(robot1.mount, robot2.mount))


@dataclass(frozen=True, eq=False)
class AttachmentSpec(AddonSpec):
    """A specification for an attachment to a robot."""
    site: str = DEFAULT_ATTACHMENT_SITE_NAME  # duplicate of `AddonSpec.site` with a different default value.

    @property
    def _asset_type(self) -> AssetType:
        return AssetType.ATTACHMENT


@dataclass(frozen=True, eq=False)
class MountSpec(AddonSpec):
    """A specification for an asset on which to mount a robot."""

    @property
    def _asset_type(self) -> AssetType:
        return AssetType.MOUNT
