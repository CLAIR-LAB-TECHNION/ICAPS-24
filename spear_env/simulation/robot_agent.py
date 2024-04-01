from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from .entity import Entity
from ..common import InfoDict
from ..episode.specs.camera_spec import CameraSpec
from ..rendering import OffscreenRenderer

if TYPE_CHECKING:
    from .simulator import Simulator


class RobotAgent:
    """
    Represents an acting agent corresponding to a robot in the simulation.

    public fields:
        - spec:
            The robot specification object with which the robot model was loaded.
        - entity:
            An `Entity` object (see `spear_env.simulation.entity.Entity`) binding the robot model to the simulation
            physics.
        - sensor_map:
            A dictionary (str --> Sensor) where Sensor is a `spear_env.simulation.Entity` object representing a sensor
            in the model. The keys are the sensor type and the identifiers of the sensors in the model, delimited by two
            colon symbols. For example, the key for a force sensor whose identifier is "my_robot/sensor1" will be
            "force::my_robot/sensor1".
        - actuators:
            An entity `Entity` object (see `spear_env.simulation.entity.Entity`) binding the list of actuator elements
            in the model over which the agent has control.
    """

    def __init__(self, sim: Simulator):
        """
        Creates a new agent object for the simulation robot.
        :param sim: The simulation in which the robot resides.
        """
        # TODO agent spec to determine agent sensors and actuators instead of robot spec

        self.spec = sim.robot
        self.entity = Entity(sim.composer.get_mounted_robot(self.spec), sim.physics)
        self.sensor_map = {
            f'{sensor.element_tag}::{sensor.identifier}': sensor
            for sensor in Entity.from_list(sim.mjcf_model.find_all('sensor'), sim.physics)
        }
        self.camera_map = {
            name: renderer
            for name, renderer in [
                self.__get_camera_name_and_renderer(cam_spec, sim)
                for cam_spec in self.spec.cameras
            ]
        }
        self.actuators = Entity(sim.composer.mjcf_model.find_all('actuator'), sim.physics)

    def set_action(self, ctrl: npt.NDArray[np.float64]) -> None:
        """
        set the control values for the actuators controlled by the agent.
        :param ctrl: the control values to set
        :return:
        """
        self.actuators.ctrl[:] = ctrl

    def reset(self):
        """resets the joint positions of the robot"""
        self.entity.configure_joints(position=self.spec.init_pos, velocity=self.spec.init_vel)

    @property
    def observation_space(self) -> gym.core.spaces.Space:
        """
        The observation space of the agent as per the `gymnasium.spaces` standard. This is the space of possible joint
        positions and velocities.
        """

        # get lengths of joiint position an velocity
        pos_len = len(self.entity.get_joint_positions())
        vel_len = len(self.entity.get_joint_velocities())

        # calculate the position and velocity bounds (lows and highs) for each joint
        pos_bounds = self.entity.get_joint_ranges()
        vel_bounds = [[-np.inf, np.inf]] * vel_len

        # merge the position and velocity bounds and split to lower and upper bound lists
        state_lows, state_highs = np.concatenate([pos_bounds, vel_bounds]).T

        # set robot_joint_pos observation space
        spaces = dict(
            robot_state=gym.spaces.Box(low=state_lows, high=state_highs, shape=(pos_len + vel_len,), dtype=np.float64)
        )

        # set sensor observation spaces
        spaces.update({
            name: sensor.sensordata_space
            for name, sensor in self.sensor_map.items()
        })

        # merge to dictionary observation
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self) -> gym.spaces.Space:
        """The action space of the agent as per the `gymnasium.spaces` standard"""

        # get control bounds for actuators
        bounds = self.actuators.ctrlrange.copy().astype(float)

        # get control limits for actuators
        limits = self.actuators.ctrllimited.copy().astype(float)

        # remove constrained bounds for unlimited actuators
        bounds[limits == 0] = [-np.inf, np.inf]

        # split to lower and upper bound lists
        low, high = bounds.T

        return gym.spaces.Box(low=low, high=high, dtype=np.float64)

    def get_obs(self) -> dict[str, npt.NDArray[np.float64]]:
        """
        Retrieves the current agent observation.
        :return: A dictionary of agent observations.
        """
        # get agent state
        out = dict(
            robot_state=np.concatenate([self.entity.get_joint_positions(), self.entity.get_joint_velocities()])
        )

        # get sensor data
        out.update({
            name: sensor.sensordata
            for name, sensor in self.sensor_map.items()
        })

        out.update({
            name: renderer.render()
            for name, renderer in self.camera_map.items()
        })

        return out

    def get_info(self) -> InfoDict:
        """
        retrieves an agent-specific information dictionary.
        :return: a dictionary of agent information.
        """
        return dict(
            qpos=self.entity.get_joint_positions(),
            qvel=self.entity.get_joint_velocities(),
        )

    @staticmethod
    def __get_camera_name_and_renderer(cam_spec: CameraSpec, sim: Simulator):
        identifier = Entity.from_name_and_tag(cam_spec.identifier, 'camera', sim.mjcf_model, sim.physics).identifier

        renderer = OffscreenRenderer(sim.model, sim.data, identifier, width=cam_spec.width, height=cam_spec.height,
                                     depth=cam_spec.depth, segmentation=cam_spec.segmentation)

        dims = f'{renderer.viewer.height}X{renderer.viewer.width}'
        img_type = 'segmentation' if cam_spec.segmentation else 'depth' if cam_spec.depth else 'rgb'

        name = f'camera_{identifier}_{dims}_{img_type}'

        return name, renderer
