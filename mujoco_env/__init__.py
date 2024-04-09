from . import milestones  # registers the milestones
from gymnasium import make
from .common import cfg_keys
from .episode import *
from .mujoco_env import MujocoEnv
from .tasks.rearrangement.rearrangement_task import COMRearrangementTask as _COMTask

__all__ = [
    'make',
    'MujocoEnv',
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

from_cfg = MujocoEnv.from_cfg
from_cfg_file = MujocoEnv.from_cfg_file
