""" a wrapper around spear env to simplify and fix some issues with the environment """

import numpy as np
import spear_env
from spear_env.tasks.null_task import NullTask
from block_world.PID_controller import PIDController
from block_world.grasp_manager import GraspManager


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

# for PID controller that failed :(
# was close to good with frame_skip=5
# kp = [200, 300, 250, 50, 50, 50]
# kd = [40, 50, 40, 16, 16, 16]
# ki = [5, 15, 8, 3, 3, 3]

# frame skip 1
# kp = [1500, 1500, 1500, 300, 300, 300]
# kd = [200, 200, 200, 50, 50, 50]
# ki = [20, 100, 100, 20, 20, 20]
# code PID might work with shorter timestep...


frame_skip = 5

class BlockWorld():
    def __init__(self):
        self._env = spear_env.from_cfg(cfg=env_cfg, render_mode="human", frame_skip=frame_skip)
        obs, info = self._env.reset()  # once, for info, later again
        self._mj_model = info['priveleged']['model']
        self._mj_data = info['priveleged']['data']
        self._env_entity = self._env.agent.entity

        self.robot_joint_pos = None  # will be updated in reset
        self.robot_joint_velocities = None  # --""--
        self.gripper_state_closed = False  # --""--
        self.max_joint_velocities = INIT_MAX_VELOCITY

        self._grasp_manager = GraspManager(self._mj_model, self._mj_data)

        # dt = self._mj_model.opt.timestep * frame_skip
        # self._pid_controller = PIDController(kp, ki, kd, dt)

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
        # if reset_pid:
        #     self._pid_controller.reset_endpoint(target_joint_pos)

        self._env_step(target_joint_pos, gripper_closed)
        self._clip_joint_velocities()
        return self.get_state()

    def get_state(self):
        return self.robot_joint_pos

    def get_object_pos(self, name: str):
        return self._mj_model.body(name).pos

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
        while np.linalg.norm(self.robot_joint_pos - target_joint_pos) > tolerance\
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
        # joint_control = self._pid_controller.control(self.robot_joint_pos)
        action = np.concatenate((target_joint_pos, [int(gripper_closed)]))
        obs, r, term, trunc, info = self._env.step(action)

        self.robot_joint_pos = obs['robot_state'][:6]
        self.robot_joint_velocities = obs['robot_state'][6:12]
        self.gripper_state_closed = gripper_closed

        self._env.render()


