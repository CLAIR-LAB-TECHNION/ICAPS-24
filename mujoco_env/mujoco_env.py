from __future__ import annotations

from typing import Optional, SupportsFloat, Any, Literal

import numpy as np
from gymnasium import Env
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.utils import EzPickle

from .common.defs.types import InfoDict, Config, FilePath
from .episode.samplers import EpisodeSampler, CfgEpisodeSampler, CfgFileEpisodeSampler
from .episode.specs.episode_spec import EpisodeSpec
from .simulation.robot_agent import RobotAgent
from .simulation.simulator import Simulator
from .tasks.task import Task
from .rendering import BaseRenderer, WindowRenderer, OffscreenRenderer


class MujocoEnv(Env, EzPickle):
    metadata = {
        'render_modes': [
            'human',
            'rgb_array',
            'depth_array',
            'segmentation',
        ]
        # render_fps is added in `__initialize_renderer`.
        # FPS may change if the simulation is reinitialized at runtime.
    }

    def __init__(
            self,
            episode_sampler: EpisodeSampler,
            render_mode: Optional[Literal['human', 'rgb_array', 'segmentation', 'depth_array']] = None,
            frame_skip: int = 1,
            reset_on_init: bool = True
    ) -> None:

        # easy reconstruction of pickled environments via the constructor
        EzPickle.__init__(self, episode_sampler, frame_skip, render_mode, reset_on_init)

        # set _env parameters
        self.episode_sampler = episode_sampler
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        # declare empty _env variables. will be set upon environment reset
        self.episode: Optional[EpisodeSpec] = None
        self.sim: Optional[Simulator] = None
        self.agent: Optional[RobotAgent] = None
        self.task: Optional[Task] = None
        self.renderer: Optional[BaseRenderer] = None

        # provide dummy observation and action spaces for gym compatibility.
        # on reset, these will change to that one matching the sampled episode.
        self.observation_space = spaces.Box(np.inf, -np.inf, (0,))
        self.action_space = spaces.Box(np.inf, -np.inf, (0,))

        # reset _env if specified.
        # this will sample an episode and load the simulation
        if reset_on_init:
            self.reset()

    # ==============================
    # ========== Init API ==========
    # ==============================

    @classmethod
    def from_cfg(
            cls,
            cfg: Config,
            render_mode: Optional[Literal['human', 'rgb_array', 'segmentation', 'depth_array']] = None,
            frame_skip: int = 1,
            reset_on_init: bool = True
    ) -> MujocoEnv:
        return cls(CfgEpisodeSampler(cfg), render_mode, frame_skip, reset_on_init)

    @classmethod
    def from_cfg_file(
            cls,
            cfg_file: FilePath,
            render_mode: Optional[Literal['human', 'rgb_array', 'segmentation', 'depth_array']] = None,
            frame_skip: int = 1,
            reset_on_init: bool = True
    ):
        return cls(CfgFileEpisodeSampler(cfg_file), render_mode, frame_skip, reset_on_init)

    # ===================================
    # ========== Gymnasium API ==========
    # ===================================

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # run task pre-step callback
        self.task.begin_frame(action)

        # step simulation
        self.do_simulation(action, self.frame_skip)

        # run task post-step callback
        self.task.end_frame(action)

        return (
            self.agent.get_obs(),  # get agent observation
            self.task.score(),  # get task reward
            self.task.is_done(),  # check if task is done
            False,  # _env is never truncated internally
            self.__get_info_dict()  # get info dict
        )

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # sample episode and load simulation if needed
        self.__set_next_episode()

        with self.sim.physics.reset_context():
            # reset the simulation to the (optionally provided) keyframe in the scene specification.
            # keyframe data is overridden by init_pos and init_vel.
            self.sim.reset()

            # reset the agent state in the simulation (overrides keyframe data)
            self.agent.reset()

            # reset task to episode parameters (overrides keyframe data and agent state)
            # with self.sim.physics.reset_context():
            self.task.reset(**self.episode.task.params)

        return self.agent.get_obs(), self.__get_info_dict()

    def render(self):
        if self.render_mode is None:
            raise AttributeError(f'Cannot render environment without setting `render_mode`. set to one of: '
                                 f'{self.metadata["render_modes"]}')

        # TODO follow github issue for a solution to this with the passive viewer
        #  https://github.com/google-deepmind/mujoco/issues/1082
        # if self.render_mode == 'human':
        #     try:
        #         self.task.update_render(self.renderer._get_viewer(render_mode=self.render_mode))
        #     except AttributeError:
        #         pass  # task is not yet initialized

        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        if self.sim is not None:
            self.sim.free()
            self.sim = None

    # =============================
    # ========== Sim API ==========
    # =============================

    @property
    def dt(self):
        return self.sim.physics.timestep() * self.frame_skip

    def set_episode(self, episode: EpisodeSpec):
        # set new episode. keep previous episode for comparison
        self.episode, episode = episode, self.episode

        # check requirement for new model
        if episode is None or self.sim is None or EpisodeSpec.require_different_models(self.episode, episode):
            # new model required: first episode or no simulator or scene/robot mismatch.
            # reinitialize simulation
            self.initialize_simulation()
        else:
            # no new model required. swap new specifications
            self.sim.swap_specs(self.episode.scene, self.episode.robot)

        # check requirement for new task
        if episode is None or self.task is None or episode.task.cls != self.episode.task.cls:
            # new task required: first episode or task class mismatch. reinitialize task
            self.task = self.episode.task.cls(self.sim)

        # extract agent from simulation
        self.agent = self.sim.get_agent()
        self.observation_space = self.agent.observation_space
        self.action_space = self.agent.action_space

    def initialize_simulation(self):
        if self.sim is None:
            # initialize a new simulation
            self.sim = Simulator(self.episode.scene, self.episode.robot)
        else:
            # swap to new specifications.
            self.sim.swap_specs(self.episode.scene, self.episode.robot)

        # initialize parameters for rendering
        self.__initialize_renderer()

    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError(
                f"Action dimension mismatch. Expected {self.action_space.shape}, found {np.array(ctrl).shape}"
            )

        # set action in agent
        self.agent.set_action(ctrl)

        # step simulation desired number of frames
        self.sim.step(n_frames)

    # ================================== #
    # ========== init helpers ========== #
    # ================================== #

    def __set_next_episode(self):
        new_episode = self.episode_sampler.sample()
        self.set_episode(new_episode)

    def __set_observation_space(self):
        # empty dict not allowed. use empty box
        self.observation_space = self.agent.observation_space

    def __set_action_space(self):
        bounds = self.sim.model.actuator_ctrlrange.copy().astype(np.float64)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)
        return self.action_space

    def __initialize_renderer(self):
        # set metadata FPS according to the timestep actual clock time
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        # close previous renderer from memory (if exists)
        if self.renderer is not None:
            self.renderer.close()

        # initialize mujoco renderer
        if self.render_mode == 'human':
            self.renderer = WindowRenderer(self.sim.model, self.sim.data, self.episode.scene.render_camera,
                                           render_fps=self.metadata["render_fps"],
                                           **self.episode.scene.renderer_cfg)
        else:
            self.renderer = OffscreenRenderer(self.sim.model, self.sim.data, self.episode.scene.render_camera,
                                              depth=self.render_mode == 'depth_array',
                                              segmentation=self.render_mode == 'segmentation',
                                              **self.episode.scene.renderer_cfg)

    # ==================================
    # ========== Info Helpers ==========
    # ==================================

    def __get_info_dict(self) -> InfoDict:
        return dict(
            task=self.task.get_info(),
            agent=self.agent.get_info(),
            priveleged=self.sim.get_privileged_info() if self.episode.robot.privileged_info else {}
        )
