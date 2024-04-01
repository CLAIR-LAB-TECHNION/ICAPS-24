from __future__ import annotations

import os
from pathlib import Path

from .defs.types import FilePath, AssetType

# The path to the directory containing all internal assets provided by the library
ASSETS_DIR = Path(__file__).parent.parent / 'assets'


def get_internal_asset_file_path(p: FilePath, asset_type: AssetType | str) -> FilePath:
    """
    determine if the given path is an asset name or a path to an asset file paths. asset names are converted to asset
    file paths for internally named assets.
    :param p: the path or asset name
    :param asset_type: the type of asset (scene, robot, task)
    :return:
    """
    # path was given
    if isinstance(p, os.PathLike):
        return p

    # usage of path separator hints that this is a path
    if os.sep in p:
        return Path(p)

    # verify asset_type input
    if not isinstance(asset_type, AssetType):
        asset_type = AssetType(asset_type)  # will raise an ERROR if the given string is not a valid asset type value

    # get the correct file extension for this given asset type
    if asset_type == AssetType.EPISODE:
        ext = 'yml'
    else:
        ext = 'xml'

    # assemble absolute path to the asset file
    return ASSETS_DIR / f'{asset_type.value}s' / p / f'{asset_type.value}.{ext}'
