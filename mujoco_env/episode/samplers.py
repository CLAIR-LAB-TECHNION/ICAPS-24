from abc import ABC, abstractmethod
from typing import Optional, Any, Type, Sequence, Collection

import numpy as np
import yaml

from .cfg_parsing import episode_from_cfg
from .specs.episode_spec import EpisodeSpec
from .specs.robot_spec import RobotSpec
from .specs.scene_spec import SceneSpec
from .specs.task_spec import TaskSpec
from ..common.assets import get_internal_asset_file_path
from ..common.defs.cfg_keys import GLOBALS, EPISODES
from ..common.defs.types import FilePath, ParamsDict, AssetType, Config
from ..common.misc import nested_dict_update
from ..tasks.task import Task

__all__ = [
    'EpisodeSampler',
    'SingleEpisodeSampler',
    'MultiTaskEpisodeSampler',
    'MultiParamTaskEpisodeSampler',
    'MultiRobotEpisodeSampler',
    'FiniteRandomEpisodeSampler',
    'CfgEpisodeSampler',
    'CfgFileEpisodeSampler',
]


class EpisodeSampler(ABC):
    """An interface for sampling episode specifications."""

    @abstractmethod
    def sample(self) -> EpisodeSpec:
        """
        samples a single episode specification.
        :return: an episode specification object.
        """
        pass


class SingleEpisodeSampler(EpisodeSampler):
    """An episode sampler for a single episode."""

    def __init__(self, episode: EpisodeSpec) -> None:
        """
        Creates a new sampler for a single episode.
        :param episode: the single episode to be constantly sampled.
        """
        self.episode = episode

    def sample(self) -> EpisodeSpec:
        return self.episode

    @classmethod
    def from_individual_specs(cls, scene: SceneSpec, robot: RobotSpec, task: TaskSpec) -> EpisodeSampler:
        """
        An initialization helper using the internal specs in an episode.
        :param scene: The scene specification.
        :param robot: The robot specification.
        :param task: The task specification.
        :return: An instance of this class that samples an episode with the given scene, robot, and task.
        """
        return cls(EpisodeSpec(scene, robot, task))


class MultiTaskEpisodeSampler(EpisodeSampler, ABC):
    """An episode sampler for changing tasks and a constant scene and robot."""

    def __init__(self, scene: SceneSpec, robot: RobotSpec) -> None:
        """
        Creates a new sampler for a single scene-robot pair.
        :param scene: The scene specification.
        :param robot: The robot specification.
        """
        self.scene = scene
        self.robot = robot

    def sample(self) -> EpisodeSpec:
        return EpisodeSpec(scene=self.scene,
                           robot=self.robot,
                           task=self._sample_task())

    @abstractmethod
    def _sample_task(self) -> TaskSpec:
        """
        Samples the task specification for the next sampled episode.
        :return: A task specification object.
        """
        pass


class MultiParamTaskEpisodeSampler(MultiTaskEpisodeSampler, ABC):
    """An episode sampler for changing task parameters and constant scene, robot, and task"""

    def __init__(self, scene: SceneSpec, robot: RobotSpec, task_cls: Type[Task]) -> None:
        """
        Creates a sampler for a single scene-robot-task triplet with changing task parameters.
        :param scene: The scene specification.
        :param robot: The robot specification.
        :param task_cls: The class of the task to be a part of all sampled episode specifications.
        """
        super().__init__(scene, robot)
        self.task_cls = task_cls

    def _sample_task(self) -> TaskSpec:
        return TaskSpec(cls=self.task_cls, params=self._sample_task_params())

    @abstractmethod
    def _sample_task_params(self) -> ParamsDict:
        """
        Samples a set of parameters that will be given to the task upon environment reset.
        :return: A dictionary (str --> Any) of parameters.
        """
        pass


class MultiRobotEpisodeSampler(EpisodeSampler, ABC):
    """An episode sampler for changing tasks and a constant scene and robot."""

    def __init__(self, scene: SceneSpec, task: TaskSpec) -> None:
        """
        Creates a new sampler for a single scene-task pair.
        :param scene: The robot specification.
        :param task: The task specification.
        """
        self.scene = scene
        self.task = task

    def sample(self) -> EpisodeSpec:
        return EpisodeSpec(scene=self.scene,
                           robot=self._sample_robot(),
                           task=self.task)

    @abstractmethod
    def _sample_robot(self) -> RobotSpec:
        """
        Samples the robot specification for the next sampled episode.
        :return: A robot specification object.
        """
        pass


class FiniteRandomEpisodeSampler(EpisodeSampler):
    """An episode sampler for a finite collection of episodes"""

    def __init__(self,
                 episodes: Sequence[EpisodeSpec],
                 p: Optional[Sequence[float]] = None) -> None:
        """
        Creates a new sampler that samples from a finite collection of episodes.
        :param episodes: a finite collection of episode specifications.
        :param p: a collection of probability weights for sampling bias.
        """
        if p is not None:
            assert len(episodes) == len(p), 'number of choices must much the number of probability weights'

        self.episodes = episodes
        self.p = p

    def sample(self) -> EpisodeSpec:
        sample_i = np.random.choice(self.num_episodes, p=self.p)
        return self.episodes[sample_i]

    @property
    def num_episodes(self) -> int:
        """The number of items in the collection of episodes from which we sample."""
        return len(self.episodes)


class CfgEpisodeSampler(FiniteRandomEpisodeSampler):
    """
    An episode sampler for a finite collection of episodes specified in a configuration API.
    A sampler configuration is a dictionary that describes a finite collection of episodes. The API for such a sampler
    is described as follows:
    - top level keys: The `globals` and `episodes` keys are the top level keys (see `spear_env.common.defs.cfg_keys`)
                      for episode sampler configurations.
    - `episodes` key: refers to a one or a list of episode configurations
    - `globals` key (optional): a partial episode configuration that defines global defaults for all episodes that will
                                be overridden by the actual episode configurations themselves.
    - internal configurations:  configurations for defining internal components of the episode. see
                                `spear_env.episode.cfg_parsing.episode_from_cfg` for more details.

    >>> from mujoco_env import cfg_keys
    >>> global_cfg = {  # high-level defaults
    ...     cfg_keys.SCENE: 'floatworld',
    ...     cfg_keys.ROBOT: {
    ...         cfg_keys.RESOURCE: 'ur5e',
    ...         cfg_keys.ROBOT_ATTACHMENTS: 'adhesive_gripper'
    ...     },
    ...     cfg_keys.TASK: 'spear_env.tasks:NullTask'
    ... }
    >>> cfg = {
    ...     cfg_keys.GLOBALS: {  # low-level defaults (overrides high-level defaults)
    ...         cfg_keys.SCENE: {  # completely override high-level defaults scene
    ...             cfg_keys.RESOURCE: 'flooorworld',
    ...             cfg_keys.SCENE_OBJECTS: ['milk', 'bread']
    ...         }
    ...     },
    ...     cfg_keys.EPISODES: [  # actual episode configurations (overrides low-level defaults)
    ...         {
    ...             cfg_keys.ROBOT: {
    ...                 cfg_keys.RESOURCE: 'ur5e',
    ...                 cfg_keys.ROBOT_BASE_POS: [1, 0, 0]  # added new field
    ...             }
    ...         },
    ...         {
    ...             'robot': {  # don't have to use `cfg_keys`
    ...                 'resource': 'ur5e',
    ...                 'base_pos': [0, 1, 0],
    ...                 'attachments': None  # override high-level attachments default
    ...             }
    ...         }
    ...     ]
    ... }
    >>> sampler = CfgEpisodeSampler(cfg, global_cfg)
    >>> sampler.episodes  # doctest:+ELLIPSIS
    [EpisodeSpec(...), ..., EpisodeSpec(...)]
    >>> sampler.num_episodes
    2
    """

    def __init__(self, cfg: Config, global_cfg: Config = None):
        """
        Creates a new sampler from a configuration dictionary.
        :param cfg: a configuration dictionary according to the described API in the class docstring. if provided
                    episodes is a dictionary, assume it is a configuration for a single episode. If no episodes
                    key is provided, assume the entire input dictionary defines a single episode with no global
                    defaults.
        :param global_cfg: a high-level global default configuration dictionary. This is overriden by a global
                           configuration if provided in `cfg`.
        """

        if EPISODES in cfg:
            episode_cfgs = cfg[EPISODES]
        else:
            episode_cfgs = cfg  # assume entire cfg is episode cfgs

        if isinstance(episode_cfgs, dict):
            episode_cfgs = [episode_cfgs]  # assume cfg is a single episode

        if global_cfg is None:  # globals not given, but may be in cfg
            global_cfg = cfg.pop(GLOBALS, {})
        elif GLOBALS in cfg:  # globals given in cfg dict and function arg. cfg dict globals take precedence
            nested_dict_update(src=cfg[GLOBALS], dest=global_cfg, inplace=True)

        # iterate episode cfgs and update with globals
        episodes = []
        for cfg in episode_cfgs:
            nested_dict_update(src=global_cfg, dest=cfg, inplace=True)  # update with globals
            episode = episode_from_cfg(cfg)  # create episode
            episodes.append(episode)  # add to list

        # initialize finite sampler
        super().__init__(episodes, p=cfg.get('sample_weights'))


class CfgFileEpisodeSampler(CfgEpisodeSampler):
    """
    An episode sampler for a finite collection of episodes specified in a configuration YAML file according to the API
    defined in the `CfgEpisodeSampler` (the super-class).
    """
    def __init__(self, cfg_file: FilePath, lazy=False):
        """
        Creates a new sampler based on the contents of the given configuration file
        :param cfg_file: A YAML file containing an episode sampler configuration according to the API in the
                         `CfgEpisodeSampler` super-class.
        :param lazy: If `True`, the sampler does not read or parse the configuration file until `sample` is called for
                     the first time. Before this occurs, fields and properties may contain incorrect values.
        """
        # allow giving predefined episode configuration assets
        self.cfg_file = get_internal_asset_file_path(cfg_file, AssetType.EPISODE)

        self.lazy = lazy
        self.is_init = False

        if self.lazy:
            super().__init__({EPISODES: []})
        else:
            self._load_cfg()

    def _load_cfg(self):
        with open(self.cfg_file, 'r') as f:
            cfg = yaml.full_load(f)  # full load for python object support

        super().__init__(cfg)
        self.is_init = True

    def sample(self) -> EpisodeSpec:
        if not self.is_init:
            self._load_cfg()

        return super().sample()
