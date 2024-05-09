""" a wrapper around spear env to simplify and fix some issues with the environment """
from copy import deepcopy

import numpy as np
import mujoco_env
from n_table_blocks_world.PID_controller import PIDController
from n_table_blocks_world.grasp_manager import GraspManager
from n_table_blocks_world.object_manager import ObjectManager
from n_table_blocks_world.configurations_and_constants import *


class NTableBlocksWorld():
    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self._env = mujoco_env.from_cfg(cfg=env_cfg, render_mode=render_mode, frame_skip=frame_skip)
        obs, info = self._env.reset()  # once, for info, later again
        self._mj_model = info['priveleged']['model']
        self._mj_data = info['priveleged']['data']
        self._env_entity = self._env.agent.entity

        self.robot_joint_pos = None  # will be updated in reset
        self.robot_joint_velocities = None  # --""--
        self.gripper_state_closed = False  # --""--
        self.max_joint_velocities = INIT_MAX_VELOCITY

        self._object_manager = ObjectManager(self._mj_model, self._mj_data)
        self._grasp_manager = GraspManager(self._mj_model, self._mj_data, self._object_manager, min_grasp_distance=0.1)

        self._ee_mj_data = self._mj_data.body('rethink_mount_stationary/ur5e/adhesive gripper/')
        # dt = self._mj_model.opt.timestep * frame_skip
        # self._pid_controller = PIDController(kp, ki, kd, dt)

        self.reset()

    def reset(self):
        self.max_joint_velocities = INIT_MAX_VELOCITY

        obs, _ = self._env.reset()
        self._env_entity.set_state(position=INIT_CONFIG)
        self.robot_joint_pos = INIT_CONFIG
        self.robot_joint_velocities = obs["robot_state"][6:12]
        self.gripper_state_closed = False
        self._grasp_manager.release_object()
        self._object_manager.reset_object_positions()

        self.step(INIT_CONFIG, gripper_closed=False)

        if self.render_mode == "human":
            self._env.render()

        return self.get_state()

    def step(self, target_joint_pos, gripper_closed=None):
        # if reset_pid:
        #     self._pid_controller.reset_endpoint(target_joint_pos)
        if gripper_closed is None:
            gripper_closed = self.gripper_state_closed

        self._env_step(target_joint_pos, gripper_closed)
        self._clip_joint_velocities()

        if gripper_closed:
            if self._grasp_manager.attatched_object_name is not None:
                self._grasp_manager.update_grasped_object_pose()
            else:
                self._grasp_manager.grasp_nearest_object_if_close_enough()
        else:
            self._grasp_manager.release_object()

        if self.render_mode == "human":
            self._env.render()

        return self.get_state()

    def simulate_steps(self, n_steps):
        """
        simulate n_steps in the environment without moving the robot
        """
        config = self.robot_joint_pos
        for _ in range(n_steps):
            self.step(config)

    def render(self):
        return self._env.render()

    def get_state(self):
        object_positions = self._object_manager.get_all_object_positons_dict()
        state = {"robot_joint_pos": self.robot_joint_pos,
                 "robot_joint_velocities": self.robot_joint_velocities,
                 "gripper_state_closed": self.gripper_state_closed,
                 "object_positions": object_positions}

        return deepcopy(state)

    def get_object_pos(self, name: str):
        return self._object_manager.get_object_pos(name)

    def move_to(self, target_joint_pos, tolerance=0.05, end_vel=0.1, max_steps=None):
        """
        move robot joints to target config, until it is close within tolerance, or max_steps exceeded.
        @param target_joint_pos: position to move to
        @param tolerance: distance withing configuration space to target to consider as reached
        @param max_steps: maximum steps to take before stopping
        @return: state, success

        success is true if reached goal within tolerance, false otherwise
        """
        # self._pid_controller.reset_endpoint(target_joint_pos)

        step = 0
        while np.linalg.norm(self.robot_joint_pos - target_joint_pos) > tolerance \
                or np.linalg.norm(self.robot_joint_velocities) > end_vel:
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

        # joint_control = self._pid_controller.control(self.robot_joint_pos)  # would be relevant if we change to force
        # control and use PID controller, but we are using position control right now.

        # gripper control for the environment, which is the last element in the control vector is completely
        # ignored right now, instead we attach the nearest graspable object to the end effector and maintain it
        # with the grasp manager, outside the scope of this method.

        # action = np.concatenate((target_joint_pos, [int(gripper_closed)])) # would have been if grasping worked
        action = np.concatenate((target_joint_pos, [0]))

        obs, r, term, trunc, info = self._env.step(action)

        self.robot_joint_pos = obs['robot_state'][:6]
        self.robot_joint_velocities = obs['robot_state'][6:12]
        self.gripper_state_closed = gripper_closed

    def get_ee_pos(self):
        return deepcopy(self._ee_mj_data.xpos)