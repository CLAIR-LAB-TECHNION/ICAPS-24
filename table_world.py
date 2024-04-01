import numpy as np
import spear_env
from spear_env.tasks.null_task import NullTask


env_cfg = dict(
    scene=dict(
        resource='tableworld',
        render_camera='top-right'
    ),
    robot=dict(
        resource='ur5e',
        mount='rethink_stationary',
        privileged_info=True,
        attachments=['adhesive_gripper'],
    ),
    task=NullTask,
)


INIT_CONFIG = np.array([0, -1.57, 0, 0, 0, 0])
INIT_MAX_VELOCITY = np.array([2, 2, 2, 2, 2, 2])

class TableWorld():
    def __init__(self):
        self._env = spear_env.from_cfg(cfg=env_cfg, render_mode="human", frame_skip=5)
        obs, info = self._env.reset()  # once, for info, later again
        self._mj_model = info['priveleged']['model']
        self._mj_data = info['priveleged']['data']
        self._env_entity = self._env.agent.entity

        self.robot_joint_pos = None  # will be updated in reset
        self.robot_joint_velocities = None  # --""--
        self.gripper_state_closed = False  # --""--
        self.max_joint_velocities = INIT_MAX_VELOCITY

        self.reset()

    def reset(self):
        """ TODO """
        self.max_joint_velocities = INIT_MAX_VELOCITY

        obs, _ = self._env.reset()
        self._env_entity.set_state(position=INIT_CONFIG)
        self.robot_joint_pos = INIT_CONFIG
        self.robot_joint_velocities = obs["robot_state"][6:12]
        self.gripper_state_closed = False

        return self.get_state()

    def step(self, target_joint_pos, gripper_closed=False):
        """
        TODO
        """
        self._env_step(target_joint_pos, gripper_closed)
        self._clip_joint_velocities()
        return self.get_state()

    def get_state(self):
        return self.robot_joint_pos

    def get_object_pos(self, name: str):
        return self._mj_model.body(name).pos

    def move_to(self, target_joint_pos, tolerance=0.05, max_steps=None):
        """
        move robot joints to target config, until it is close within tolerance, or max_steps exceeded.
        @param target_joint_pos: position to move to
        @param tolerance: distance withing configuration space to target to consider as reached
        @param max_steps: maximum steps to take before stopping
        @return: state, success

        success is true if reached goal within tolerance, false otherwise
        """
        step = 0
        while np.linalg.norm(self.robot_joint_pos - target_joint_pos) > tolerance:
            if max_steps is not None and step > max_steps:
                return self.get_state(), False

            self.step(target_joint_pos, self.gripper_state_closed)
            step += 1

        return self.get_state(), True

    def set_gripper(self, closed: bool):
        """
        close/open gripper and don't change robot configuration
        @param closed: true if gripper should be closed, false otherwise
        @return: None
        """
        self._env_step(self.robot_joint_pos, closed)

    def _clip_joint_velocities(self):
        new_vel = self.robot_joint_velocities.copy()
        new_vel = np.clip(new_vel, -self.max_joint_velocities, self.max_joint_velocities)
        self._env_entity.set_state(velocity=new_vel)
        self.robot_joint_velocities = new_vel

    def _env_step(self, target_joint_pos, gripper_closed):
        """ run environment step and update state of self accordingly"""
        action = np.concatenate((target_joint_pos, [int(gripper_closed)]))
        obs, r, term, trunc, info = self._env.step(action)

        self.robot_joint_pos = obs['robot_state'][:6]
        self.robot_joint_velocities = obs['robot_state'][6:12]
        self.gripper_state_closed = gripper_closed

        self._env.render()


