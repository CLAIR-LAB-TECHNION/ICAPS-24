from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .asset_spec import AssetSpec
from .joint_spec import JointSpec
from ...common.defs.types import Pos3D, Rot3D, Vector
from ...common.misc import set_iterable_arg


@dataclass(frozen=True, eq=False)
class AddonSpec(AssetSpec, ABC):
    """
    A specification for an addon to the simulation model. It defines the fields that are common to all addons. This is
    used to initialize and configure the addon in the simulation model.

    public fields:
        - resource: the resource file path of the addon
        - site: the site to which the addon is to be attached.
        - base_pos: the position of the base of the addon relative to the site
        - base_rot: the rotation of the base of the addon relative to the site
        - base_joints: the joints that define the addon's relationship to its parent.

    static methods:
        - require_different_models: checks if changing from one addon to another requires loading a new model into the
                                    simulation.
        - collection_require_different_models: checks if changing from one list of addons to another requires loading
                                               a new model into the simulation.
    """
    site: Optional[str] = None
    base_pos: Optional[Pos3D] = None
    base_rot: Optional[Rot3D] = None
    base_joints: Sequence[JointSpec] = tuple()

    def __post_init__(self):
        super().__post_init__()

        # set base position and rotation to a numpy array
        if self.base_pos is not None:
            super().__setattr__('base_pos', np.array(self.base_pos))
        if self.base_rot is not None:
            super().__setattr__('base_rot', np.array(self.base_rot))

        # set the base joints to a tuple of JointSpecs
        super().__setattr__('base_joints', set_iterable_arg(JointSpec, self.base_joints))

    @staticmethod
    def require_different_models(addon1: Optional['AddonSpec'], addon2: Optional['AddonSpec']) -> bool:
        """
        Check if two addons require instantiating different simulation models. That is, if we were to exchange one for
        the other, check if it would be necessary to load a new model into the simulation.
        :param addon1: an addon to compare.
        :param addon2: another addon to compare.
        :return: True if the addons require different models, False otherwise.
        """
        if addon1 is addon2 is None:  # both are None. no concern about model
            return False

        return (
                addon1 is None or  # if one is None and the other isn't, need to load new model with/without attachment
                addon2 is None or
                addon1.resource != addon2.resource or  # load from different resources
                addon1.site != addon2.site or  # attach to different sites
                np.any(addon1.base_pos != addon2.base_pos) or  # attach at different positions
                np.any(addon1.base_rot != addon2.base_rot) or  #
                addon1.base_joints != addon2.base_joints)

    @classmethod
    def collection_require_different_models(cls,
                                            addons1: Sequence['AddonSpec'],
                                            addons2: Sequence['AddonSpec']) -> bool:
        """
        checks if changing from one list of addons to another requires loading a new model into the simulation.
        :param addons1: a collection of addons to compare.
        :param addons2: another collection of addons to compare.
        :return: True if the collections require different models, False otherwise.
        """
        # there cannot be a match between all addons in the two collections if they have different lengths
        if len(addons1) != len(addons2):
            return True

        # find a match for
        addons2 = list(addons2)  # convert addons to set for quick removal of matches
        for addon1 in addons1:  # iterate addons1 and find matches in addons2

            match = None  # initialize match container
            for addon2 in addons2:  # iterate addons2 and find a match for addon1
                if not cls.require_different_models(addon1, addon2):
                    match = addon2  # match found move to next addon1
                    break

            if match is None:
                return True  # match not found. addons require different models

            addons2.remove(match)  # remove match from addons2 to prevent duplicate matches

        # there is a one-to-one match between addons1 and addons2 in terms of model requirements
        return False


@dataclass(frozen=True, eq=False)
class InitStateAddonSpec(AddonSpec, ABC):
    """
    A specification for an addon to the simulation model which defines the initial state of the addon.

    public fields:
        - init_pos: the initial position of the addon's joints
        - init_vel: the initial velocity of the addon's joints
    """
    init_pos: Optional[Vector] = None
    init_vel: Optional[Vector] = None

    def __post_init__(self):
        super().__post_init__()

        # set initial position and velocity to a numpy array
        if self.init_pos is not None:
            super().__setattr__('init_pos', np.array(self.init_pos))
        if self.init_vel is not None:
            super().__setattr__('init_vel', np.array(self.init_vel))
