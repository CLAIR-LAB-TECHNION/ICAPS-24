from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .base_spec import Spec
from ...common.assets import get_internal_asset_file_path
from ...common.defs.types import Asset, AssetType


@dataclass(frozen=True, eq=False)
class AssetSpec(Spec, ABC):
    """
    A specification for an asset to be added to the simulation model.

    public fields:
        - resource: the resource file path of the asset
    """
    resource: Asset

    def __post_init__(self):
        # Turn the given input asset into an internal asset file path
        super().__setattr__('resource', get_internal_asset_file_path(self.resource, self._asset_type))

    @property
    @abstractmethod
    def _asset_type(self) -> AssetType:
        """
        The type of asset this spec is for. This is used to determine internal asset file paths.
        """
        pass
