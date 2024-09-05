""" a wrapper around spear env to simplify and fix some issues with the environment """
from copy import deepcopy

import gymjoco
from n_table_blocks_world.grasp_manager import GraspManager
from n_table_blocks_world.object_manager import ObjectManager
from n_table_blocks_world.configurations_and_constants import *
from .configurations_and_constants import ROBOTIQ_2F85_BODY, ADHESIVE_BODY
from .utils import convert_mj_struct_to_namedtuple
from collections import defaultdict


class NTableBlocksWorld:
    def __init__(self, render_mode="human", cfg=env_cfg, ee_name=ADHESIVE_BODY,
                 grasp_joints=None, grasp_offsets=None):
        self.render_mode = render_mode
        self.ee_name = ee_name
        self.grasp_joints = grasp_joints or defaultdict(lambda: 0)
        self.grasp_offsets = grasp_offsets or defaultdict(lambda: DEFAULT_GRASP_OFFSET)
        self._env = gymjoco.from_cfg(cfg=cfg, render_mode=render_mode, frame_skip=frame_skip)
        obs, info = self._env.reset()  # once, for info, later again
        self._mj_model = info['privileged']['model']
        self._mj_data = info['privileged']['data']
        self._env_entity = self._env.agent.entity
        self.robot_joint_pos = None  # will be updated in reset
        self.robot_joint_velocities = None  # --""--
        self.camera_renders = None
        self.gripper_state_closed = False  # --""--
        self.max_joint_velocities = INIT_MAX_VELOCITY
        self._object_manager = ObjectManager(self._env.sim)
        self._grasp_manager = GraspManager(self._mj_model, self._mj_data, self._object_manager, min_grasp_distance=0.2,
                                           ee_name=ee_name)
        self._ee_mj_data = self._mj_data.body(ee_name)

        self.__vel_size = len(self._env_entity.get_joint_velocities())

        # Collision avoidance for gripper
        # self._avoid_gripper_collisions()

        if ee_name == ROBOTIQ_2F85_BODY:
            self.gripper_body = self._env.sim.get_entity(ROBOTIQ_2F85_BODY, 'body')
            gripper_mjcf = self.gripper_body.mjcf_element
            geoms = gripper_mjcf.find_all('geom')
            for geom in geoms:
                geom_id = self._mj_model.geom(geom.full_identifier).id
                # This geom belongs to the gripper
                # Set its collision type to the gripper group
                self._mj_model.geom_contype[geom_id] = 0
                # Clear its collision affinity (it won't collide with anything)
                self._mj_model.geom_conaffinity[geom_id] = 0

        # dt = self._mj_model.opt.timestep * frame_skip
        # self._pid_controller = PIDController(kp, ki, kd, dt)

        self.reset()

    def reset(self):
        self.max_joint_velocities = INIT_MAX_VELOCITY

        obs, _ = self._env.reset()
        robot_vel_start_idx = len(obs["robot_state"])
        self.robot_joint_pos = obs['robot_state'][:6]
        self.robot_joint_velocities = obs["robot_state"][robot_vel_start_idx:robot_vel_start_idx + 6]
        self.gripper_state_closed = False
        self._grasp_manager.release_object()
        self._object_manager.reset_object_positions()

        self.step(self.robot_joint_pos, gripper_closed=False)

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
                grasp_offset = self.grasp_offsets[self._grasp_manager.attatched_object_name]
                self._grasp_manager.update_grasped_object_pose(grasp_offset)
                if self.ee_name == ROBOTIQ_2F85_BODY:
                    grasp_pincher_pos = self.grasp_joints[self._grasp_manager.attatched_object_name]
                    self.gripper_body.configure_joints(position=[0, 0, grasp_pincher_pos, 0, 0, 0, grasp_pincher_pos, 0])
            else:
                self._grasp_manager.grasp_nearest_object_if_close_enough()
        else:
            self._grasp_manager.release_object()
            if self.ee_name == ROBOTIQ_2F85_BODY:
                self.gripper_body.configure_joints(position=[0, 0, 0, 0, 0, 0, 0, 0])

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
        if self._env.render_mode is not None:
            return self._env.render()

        return None

    def get_state(self):
        object_positions = self._object_manager.get_all_object_positons_dict()
        state = {"robot_joint_pos": self.robot_joint_pos,
                 "robot_joint_velocities": self.robot_joint_velocities,
                 "gripper_state_closed": self.gripper_state_closed,
                 "object_positions": object_positions,
                 "grasped_object": self._grasp_manager.attatched_object_name,
                 "geom_contact": convert_mj_struct_to_namedtuple(self._env.sim.data.contact),
                 "camera_renders": self.camera_renders}

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
        new_vel = np.concatenate((new_vel, np.zeros(self.__vel_size - len(new_vel))))
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

        robot_vel_start_idx = len(obs['robot_state']) // 2
        self.robot_joint_pos = obs['robot_state'][:6]
        self.robot_joint_velocities = obs['robot_state'][robot_vel_start_idx:robot_vel_start_idx+6]
        self.camera_renders = {k: obs[k] for k in obs if k.startswith('camera')}
        self.gripper_state_closed = gripper_closed

    def get_ee_pos(self):
        return deepcopy(self._ee_mj_data.xpos)