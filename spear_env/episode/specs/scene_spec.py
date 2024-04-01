from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .asset_spec import AssetSpec
from .addon_spec import InitStateAddonSpec
from ...common.defs.types import Identifier, AssetType
from ...common.misc import set_iterable_arg


@dataclass(frozen=True, eq=False)
class SceneSpec(AssetSpec):
    """
    A specification for a scene to be simulated.

    public fields:
        - objects: A collection of objects to be added to the scene.
        - init_keyframe: An identifier to the keyframe with which to initialize the simulation state.
    """
    objects: Sequence[ObjectSpec, ...] = tuple()
    render_camera: Identifier = -1  # default to free camera
    renderer_cfg: dict = field(default_factory=dict)  #TODO move to _env init
    init_keyframe: Optional[Identifier] = None

    def __post_init__(self):
        super().__post_init__()

        # set objects to a tuple of ObjectSpecs
        super().__setattr__('objects', set_iterable_arg(ObjectSpec, self.objects))

    @property
    def _asset_type(self) -> AssetType:
        return AssetType.SCENE

    @staticmethod
    def require_different_models(scene1: 'SceneSpec', scene2: 'SceneSpec') -> bool:
        """
        Check if two scenes require instantiating different simulation models. That is, if we were to exchange one for
        the other, check if it would be necessary to load a new model into the simulation.
        :param scene1: a scene to compare.
        :param scene2: another scene to compare.
        :return: True if the scenes require different models, False otherwise.
        """
        return (scene1.resource != scene2.resource or
                ObjectSpec.collection_require_different_models(scene1.objects, scene2.objects))


@dataclass(frozen=True, eq=False)
class ObjectSpec(InitStateAddonSpec):
    """A specification for an object in the scene."""

    @property
    def _asset_type(self) -> AssetType:
        return AssetType.OBJECT
