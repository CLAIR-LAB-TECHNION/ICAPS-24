import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import WrapperSpec

from spear_env.episode import *
from spear_env.tasks.rearrangement.rearrangement_task import COMRearrangementTask

# Milestone 1:
# Floating ball (translation only)
gym.register(
    id="Milestone-1",
    entry_point="spear_env.spear_env:SpearEnv",
    kwargs=dict(
        episode_sampler=CfgFileEpisodeSampler('dummy_ballworld', lazy=True),
        frame_skip=5
    )
)

# Milestone 1.5:
# Floating brick (translation and rotation)
gym.register(
    id="Milestone-1.5",
    entry_point="spear_env.spear_env:SpearEnv",
    kwargs=dict(
        episode_sampler=CfgFileEpisodeSampler('dummy_brickworld', lazy=True),
        frame_skip=5
    )
)

# Milestone 2:
# Paddle pushes ball
gym.register(
    id="Milestone-2",
    entry_point="spear_env.spear_env:SpearEnv",
    kwargs=dict(
        episode_sampler=CfgFileEpisodeSampler('dummy_paddleworld', lazy=True),
        frame_skip=5
    )
)

# Milestone 3:
# robot arm pushes object on table
gym.register(
    id="Milestone-3",
    entry_point="spear_env.spear_env:SpearEnv",
    kwargs=dict(
        episode_sampler=CfgFileEpisodeSampler('dummy_pushworld', lazy=True),
        frame_skip=5
    )
)


class __RandomPosOnTableSampler(MultiTaskEpisodeSampler):
    def _sample_task(self) -> TaskSpec:
        start_x = np.random.uniform(-0.32, 0.32)
        start_y = np.random.uniform(-0.92, -0.28)
        goal_x = np.random.uniform(-0.32, 0.32)
        goal_y = np.random.uniform(0.28, 0.92)

        return TaskSpec(
            cls=COMRearrangementTask,
            params={
                'obj_poses': {
                    'red_box': {
                        'start_pose': [start_x, start_y, 0.72] + [1, 0, 0, 0],
                        'goal_com': [goal_x, goal_y, 0.72]
                    }
                },
                'time_limit': 1000
            })


# Milestone 4:
# robot arm picks up and place a single object in a clean environment
gym.register(
    id="Milestone-4",
    entry_point="spear_env.spear_env:SpearEnv",
    kwargs=dict(
        episode_sampler=__RandomPosOnTableSampler(scene=SceneSpec('tableworld', init_keyframe='home'),
                                                  robot=RobotSpec('ur5e',
                                                                  attachments=[AttachmentSpec('adhesive_gripper')],
                                                                  mount=MountSpec('rethink_stationary'))),
        frame_skip=5
    )
)


class __RandomClutter(MultiTaskEpisodeSampler):
    def __init__(self, scene, robot, obs_heights, obj_heights):
        super().__init__(scene, robot)
        self.obj_heights = obj_heights
        self.obs_heights = obs_heights

    def _sample_task(self) -> TaskSpec:
        params = dict(obj_poses={}, obs_poses={}, time_limit=1000)

        for name, height in self.obs_heights.items():
            params['obs_poses'][f'obstacle_{name}'] = self._sample_obj_poses(height)

        for name, height in self.obj_heights.items():
            params['obj_poses'][f'pick_object_{name}'] = self._sample_obj_poses(height)

        return TaskSpec(cls=COMRearrangementTask, params=params)

    def _sample_obj_poses(self, height):
        start_x, start_y = self._ring_sample(0.3, 0.7)

        # start_quat = _np.random.uniform(-1, 1, 4)
        # start_quat = start_quat / _np.linalg.norm(start_quat)

        goal_x, goal_y = self._ring_sample(0.3, 0.7)

        return {
            'start_pose': [start_x, start_y, height] + [1, 0, 0, 0],  # start_quat.tolist(),
            'goal_com': [goal_x, goal_y, height]
        }

    def _ring_sample(self, radius_min, radius_max):
        theta = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(radius_min, radius_max)

        return radius * np.cos(theta), radius * np.sin(theta)


# Milestone 5:
# robot arm picks up and place a single object in a cluttered environment
gym.register(
    id="Milestone-5",
    entry_point="spear_env.spear_env:SpearEnv",
    kwargs=dict(
        episode_sampler=__RandomClutter(scene=SceneSpec('clutterworld', init_keyframe='home'),
                                        robot=RobotSpec('ur5e', attachments=[AttachmentSpec('adhesive_gripper')]),
                                        obs_heights=dict(zip(range(4), [0.3, 0.3, 0.3, 0.3])),
                                        obj_heights={0: 0.05}),
        frame_skip=5
    )
)

# Milestone 6:
# robot arm picks up and place multiple objects in a cluttered environment
gym.register(
    id="Milestone-6",
    entry_point="spear_env.spear_env:SpearEnv",
    kwargs=dict(
        episode_sampler=__RandomClutter(scene=SceneSpec('multiobjworld', init_keyframe='home'),
                                        robot=RobotSpec('ur5e', attachments=[AttachmentSpec('adhesive_gripper')]),
                                        obs_heights=dict(zip(range(4), [0.3, 0.3, 0.3, 0.3])),
                                        obj_heights={'red': 0.05, 'yellow': 0.05, 'cyan': 0.05}),
        frame_skip=5
    )
)


class __RandomClutterWithTables(__RandomClutter):
    def _sample_task(self) -> TaskSpec:
        task_spec = super()._sample_task()
        params = task_spec.params

        for name, height in self.obj_heights.items():
            params['obj_poses'][f'pick_object_{name}'] = self.__sample_table_poses(height)

        return TaskSpec(cls=COMRearrangementTask, params=params)

    def _sample_obj_poses(self, height):
        start_x, start_y = self._ring_sample(0.8, 1.2)

        goal_x, goal_y = self._ring_sample(0.8, 1.2)

        return {
            'start_pose': [start_x, start_y, height] + [1, 0, 0, 0],  # start_quat.tolist(),
            'goal_com': [goal_x, goal_y, height]
        }

    def __sample_table_poses(self, height):
        start_x = np.random.uniform(-0.5, 0.5)
        start_y = np.random.uniform(-2.35, -1.65)
        goal_x = np.random.uniform(-0.5, 0.5)
        goal_y = np.random.uniform(1.65, 2.35)

        return {
            'start_pose': [start_x, start_y, height] + [1, 0, 0, 0],  # start_quat.tolist(),
            'goal_com': [goal_x, goal_y, height]
        }


# Milestone 7:
# fetch robot (arm on wheeled mount) picks up and place multiple objects in a cluttered environment
gym.register(
    id="Milestone-7",
    entry_point="spear_env.spear_env:SpearEnv",
    kwargs=dict(
        episode_sampler=__RandomClutterWithTables(scene=SceneSpec('carryworld', init_keyframe='home'),
                                                  robot=RobotSpec('fetch', base_pos=[0, 0, 0.005]),
                                                  obs_heights=dict(zip(range(4), [0.3, 0.3, 0.3, 0.3])),
                                                  obj_heights={'red': 0.76, 'yellow': 0.76, 'cyan': 0.76}),
        frame_skip=5
    )
)


class __RandomClutterMS8(MultiTaskEpisodeSampler):
    def __init__(self, scene, robot, obs_heights, obj_heights):
        super().__init__(scene, robot)
        self.obj_heights = obj_heights
        self.obs_heights = obs_heights

    def _sample_task(self) -> TaskSpec:
        params = dict(obj_poses={}, obs_poses={}, time_limit=1000)

        for name, height in self.obs_heights.items():
            params['obs_poses'][f'obstacle_{name}'] = {
                'start_pose': [2, np.random.uniform(-2, 2), height] + [1, 0, 0, 0],
                'goal_com': [0, 0, 0]
            }

        for name, heights in self.obj_heights.items():
            params['obj_poses'][f'pick_object_{name}'] = self._sample_obj_poses(*heights)

        return TaskSpec(cls=COMRearrangementTask, params=params)

    def _sample_obj_poses(self, start_height, goal_height):
        start_x, start_y = self.__ring_sample([-0.9199999570846558, 0], 0, 0.5)
        goal_x, goal_y = self.__ring_sample([3.1999998092651367, 0], 0, 0.45)

        return {
            'start_pose': [start_x, start_y, start_height] + [1, 0, 0, 0],
            'goal_com': [goal_x, goal_y, goal_height]
        }

    def __ring_sample(self, pos, rmin, rmax):
        theta = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(rmin, rmax)

        return radius * np.cos(theta) + pos[0], radius * np.sin(theta) + pos[1]


# Milestone 8:
# fetch robot (arm on wheeled mount) picks up and place multiple objects in a cluttered SPEAR environment
gym.register(
    id="Milestone-8",
    entry_point="spear_env.spear_env:SpearEnv",
    kwargs=dict(
        episode_sampler=__RandomClutterMS8(scene=SceneSpec('spear_debug_obj'),
                                           robot=RobotSpec('fetch', base_pos=[0, 0, 0]),
                                           obs_heights=dict(zip(range(4), [0.3, 0.3, 0.3, 0.3])),
                                           obj_heights={'red': (0.4346, 0.6), 'yellow': (0.4346, 0.6),
                                                        'cyan': (0.4346, 0.6)}),
        frame_skip=5
    )
)
gym.register(
    id="Milestone-1-RL",
    entry_point="spear_env.spear_env:SpearEnv",
    kwargs=dict(
        episode_sampler=CfgFileEpisodeSampler('dummy_ballworld', lazy=True),
        frame_skip=5
    ),
    additional_wrappers=(
        WrapperSpec(name='BallworldRLRewardWrapper',
                    entry_point='spear_env.utils.wrappers:BallworldRLRewardWrapper',
                    kwargs={}),
        WrapperSpec(name='BallworldRLObsWrapper',
                    entry_point='spear_env.utils.wrappers:BallworldRLObsWrapper',
                    kwargs={}),
    )
)
