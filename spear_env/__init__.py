from . import milestones  # registers the milestones
from gymnasium import make
from .common import cfg_keys
from .episode import *
from .spear_env import SpearEnv
from .tasks.rearrangement.rearrangement_task import COMRearrangementTask as _COMTask

__all__ = [
    'make',
    'SpearEnv',
    'from_cfg',
    'from_cfg_file',
    'cfg_keys',
    'EpisodeSpec',
    'SceneSpec',
    'RobotSpec',
    'TaskSpec',
    'ObjectSpec',
    'AttachmentSpec',
    'MountSpec'
]

from_cfg = SpearEnv.from_cfg
from_cfg_file = SpearEnv.from_cfg_file
