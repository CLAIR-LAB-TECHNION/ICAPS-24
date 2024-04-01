from dataclasses import dataclass, field, InitVar
from enum import Enum
from typing import Optional

from ...episode.specs import Spec
from ...common.defs.types import Identifier


class CameraType(Enum):
    RGB = 'rgb'
    DEPTH = 'depth'
    SEGMENTATION = 'segmentation'


@dataclass(frozen=True, eq=False)
class CameraSpec(Spec):
    identifier: Identifier
    type: InitVar[CameraType] = CameraType.RGB
    height: Optional[int] = None  #TODO make these required
    width: Optional[int] = None
    depth: bool = field(init=False)
    segmentation: bool = field(init=False)

    def __post_init__(self, type: CameraType):
        if isinstance(type, str):
            type = CameraType(type)

        super().__setattr__('depth', type == CameraType.DEPTH)
        super().__setattr__('segmentation', type == CameraType.SEGMENTATION)
