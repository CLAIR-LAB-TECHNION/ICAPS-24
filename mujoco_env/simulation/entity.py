from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from dm_control import mjcf
from gymnasium.spaces import Box

from .mjcf_utils import get_element_from_name_and_tag


class Entity:
    """
    An object representing an entity in the simulation. It links between an MJCF element and its being in an
    instantiated simulation using the `dm_control.mjcf.physics.Binding` interface.

    NOTE: apart from the specified fields and methods in this documentation, an entity contains all the fields and
          methods of the `dm_control.mjcf.physics.Binding` interface corresponding to the specified MJCF element.

    public fields:
        - mjcf_element:
            The MJCF element that this entity represents.

        - physics_binding:
            The binding between the MJCF element and the simulation physics.

        - root_body:
            The root body of the entity in the simulation.

        - mjcf_model:
            The MJCF model in which the entity resides.

        - identifier:
            The full identifier of the entity in the MJCF model, including scoping of attached MJCF files.

        - element_tag:
            The tag of the MJCF element that this entity represents.

        - center_of_mass:
            The center of mass of the entity in the simulation. This is calculated from the shallowest child containing
            some joint which allows the entity to move and thus change its center of mass.

    public methods:
        - from_name_and_tag:
            Get an `Entity` object bound to an element corresponding to a specified name and tag.

        - from_model:
            Get an `Entity` object bound to a root element of a model.

        - from_list:
            Get a list of `Entity` objects bound to a list of MJCF elements.

        - configure_joints:
            Configures the positions and velocities of the joints in the entity.

        - get_joint_positions:
            Gets the positions of the joints in the entity.

        - get_joint_velocities:
            Gets the velocities of the joints in the entity.

        - get_joint_ranges:
            Gets the position range of the joint in the entity.

        - configure_actuators:
            Configures the activations and controls of the actuators in the entity.

        - get_actuator_activations:
            Gets the activation values of the actuators in the entity.

        - get_actuator_controls:
            Gets the control values of the actuators in the entity.

        - set_state:
            Sets the state of the entity, including joint positions and velocities, and actuator activations and
    """

    def __init__(self, mjcf_element: mjcf.Element, physics: Optional[mjcf.Physics] = None):
        """
        Creates a new entity object.
        :param mjcf_element: The MJCF element that this entity represents.
        :param physics: The simulation physics to which the entity is bound. If None, the physics is created from the
                        MJCF model containing the element.
        """
        # an ugly hack to allow setting attributes during initialization
        self.__dict__['__initializing__'] = True

        self.mjcf_element = mjcf_element  # TODO support list of elements

        # if no physics is given, create one from the MJCF model.
        if physics is None:
            physics = mjcf.Physics.from_mjcf_model(mjcf_element.root)
        self._physics = physics

        self.physics_binding = physics.bind(mjcf_element)

        # end ugly hack
        self.__dict__['__initializing__'] = False

    def __getattr__(self, item):
        # the __getattribute__ method is called before __getattr__ and thus will get attributes that already exist, such
        # as the public fields and methods of the class.

        # define Box spaces for all attributes that are arrays with the `shape` attribute.
        if item.endswith('_space') and hasattr(self.physics_binding, item.replace('_space', '')):
            value = getattr(self.physics_binding, item.replace('_space', ''))  # get attribute without "_space" suffix
            if hasattr(value, 'shape'):  # assert the existence of the `shape` attribute
                try:
                    # try to get the min and max values and dtype of the attribute
                    info = np.iinfo(value.dtype)
                    low, high = info.min, info.max
                except ValueError:
                    # default bounds are set to [-inf, inf]
                    low, high = -np.inf, np.inf

                # return a Box space with the attribute's shape and dtype
                return Box(low=low, high=high, shape=value.shape, dtype=value.dtype)

        # check if the attribute exists in the physics binding
        elif hasattr(self.physics_binding, item):
            return getattr(self.physics_binding, item)

        # otherwise, raise an AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' for element <{self.element_tag}> object has no attribute "
                             f"'{item}'")

    def __setattr__(self, key, value):
        # during initialization, set attributes normally
        if self.__dict__['__initializing__']:
            super().__setattr__(key, value)

        # set physics binding attributes directly in the physics binding object
        elif hasattr(self.physics_binding, key) and hasattr(getattr(self.physics_binding, key), 'shape'):
            setattr(self.physics_binding, key, value)

        # cannot set _space attributes. they are determiined by the shape of the attribute they represent.
        elif key.endswith('_space') and hasattr(self.physics_binding, key.replace('_space', '')):
            raise AttributeError('Cannot set space attributes')

        # otherwise, set attributes normally
        else:
            super().__setattr__(key, value)

    # ========================================= #
    # ========== easy init functions ========== #
    # ========================================= #

    @classmethod
    def from_name_and_tag(cls, name: str, tag: str, model_mjcf: mjcf.RootElement,
                          physics: mjcf.Physics = None) -> Entity:
        """
        Get an `Entity` object bound to an element corresponding to a specified name and tag.
        :param name: the name attribute of the element to bind.
        :param tag: the XML tag of the element to bind.
        :param model_mjcf: the MJCF model in which the element resides.
        :param physics: the simulation physics to which the entity is bound. If None, the physics is created from the
                        MJCF model containing the element.
        :return: An `Entity` object that binds the retrieved element.
        """
        mjcf_element = get_element_from_name_and_tag(name, tag, model_mjcf)
        return cls(mjcf_element, physics)

    @classmethod
    def from_model(cls, model_mjcf: mjcf.RootElement, physics: mjcf.Physics = None) -> Entity:
        """
        Get an `Entity` object bound to a root element of a model.
        :param model_mjcf: the MJCF model that defines the entity.
        :param physics: the simulation physics to which the entity is bound. If None, the physics is created from the
                        MJCF model containing the element.
        :return: An `Entity` object that binds the retrieved element.
        """
        return cls(model_mjcf.worldbody, physics)

    @classmethod
    def from_list(cls, mjcf_elements: list[mjcf.Element], physics: mjcf.Physics = None) -> list[Entity]:
        """
        Get a list of `Entity` objects bound to a list of MJCF elements.
        :param mjcf_elements: the MJCF elements that define each entity to be created.
        :param physics: the simulation physics to which the entities are bound. If None, the physics is created from the
                        MJCF model containing each element.
        :return: A list of `Entity` objects that bind the retrieved elements.
        """
        return [
            cls(mjcf_element, physics)
            for mjcf_element in mjcf_elements
        ]

    # ======================================= #
    # ========== common properties ========== #
    # ======================================= #

    @property
    def root_body(self) -> mjcf.Element:
        """The root body of the entity in the simulation."""
        if self.mjcf_model.parent:
            return mjcf.get_attachment_frame(self.mjcf_model)
        else:
            return self.mjcf_model.worldbody

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """The MJCF model in which the entity resides."""
        return self.mjcf_element.root

    @property
    def identifier(self):
        """The full identifier of the entity in the MJCF model, including scoping of attached MJCF files."""
        return self.mjcf_element.full_identifier

    @property
    def element_tag(self):
        """The tag of the MJCF element that this entity represents."""
        return self.mjcf_element.tag

    @property
    def center_of_mass(self) -> float:
        """
        The center of mass of the entity in the simulation. This is calculated from the shallowest child containing
        some joint which allows the entity to move and thus change its center of mass.
        """
        # assert that this is an element that has a center of mass.
        assert hasattr(self.physics_binding, 'xpos'), f'element <{self.element_tag}> has no center of mass'

        # if this is not a body (e.g. geom), return the center of mass of the element.
        if self.element_tag != 'body':
            return self.physics_binding.xpos

        # if the element is a body, we will provide the center of mass of the shallowest joint in the body to ensure
        # that the center of mass can change.

        # get all joints in the body
        all_joints = self.mjcf_element.find_all('joint')

        # if there are no joints, this body is static and has a constant center of mass.
        if not all_joints:
            warnings.warn(f'{self.element_tag} "{self.identifier}" has no joints and will always be static')
            return self.physics_binding.xpos

        # if there are joints, but no base joints, this body has a constant center of mass. we will use the parent of
        # the shallowest body containing some joint as a proxy for the center of mass. warn here
        elif not self.mjcf_element.find_all('joint', immediate_children_only=True):
            warnings.warn(f'{self.element_tag} "{self.identifier}" has no base joints and will have the same center of '
                          f'mass so long as the model is unchanged. using parent of shallowest joint in model as a '
                          f'proxy')

        # get shallowest joint
        shallowest_joint = all_joints[0]

        # get center of mass of the shallowest joint's parent body.
        return self._physics.bind(shallowest_joint.parent).xpos

    # =================================== #
    # ========== state helpers ========== #
    # =================================== #

    def configure_joints(self, position=None, velocity=None, recursive=True) -> None:
        """
        Configures the positions and velocities of the joints in the entity.
        :param position: The positions to set. If None, the positions are not changed.
        :param velocity: The velocities to set. If None, the velocities are not changed.
        :param recursive: If `True`, the joints of all attached entities are also configured.
        """
        joints_binding = self.__get_joints_binding(recursive)

        if position is not None:
            joints_binding.qpos[:] = position
        if velocity is not None:
            joints_binding.qvel[:] = velocity

    def get_joint_positions(self, recursive=True) -> np.ndarray:
        """
        Gets the positions of the joints in the entity.
        :param recursive: If `True`, the joint positions of all attached entities are also retrieved.
        :return: An array containing the positions of the joints in the entity.
        """
        return self.__get_joints_binding(recursive).qpos

    def get_joint_velocities(self, recursive=True) -> np.ndarray:
        """
        Gets the velocities of the joints in the entity.
        :param recursive: If `True`, the joint velocities of all attached entities are also retrieved.
        :return: An array containing the velocities of the joints in the entity.
        """
        return self.__get_joints_binding(recursive).qvel

    def get_joint_ranges(self, recursive=True) -> np.ndarray:
        """
        Gets the position range of the joint in the entity.
        :param recursive: If `True`, the joint position ranges of all attached entities are also retrieved.
        :return: An array of shape (N, 2) containing the low and high range of positions of the joints in the entity.
        """
        jnt_ranges = []
        for joint_element in self.mjcf_element.find_all('joint', exclude_attachments=not recursive):
            joint = self._physics.bind(joint_element)
            if joint.limited:
                jnt_range = joint.range.copy()
            else:
                jnt_range = [-np.inf, np.inf]
            jnt_ranges.extend([jnt_range] * len(joint.qpos))

        return np.stack(jnt_ranges)

    def configure_actuators(self, act=None, ctrl=None, recursive=True) -> None:
        """
        Configures the activations and controls of the actuators in the entity.
        :param act: The activations to set. If None, the activations are not changed.
        :param ctrl: The controls to set. If None, the controls are not changed.
        :param recursive: If `True`, the actuators of all attached entities are also configured.
        """
        actuators_binding = self.__get_actuators_binding(recursive)

        if act is not None:
            actuators_binding.act[:] = act
        if ctrl is not None:
            actuators_binding.ctrl[:] = ctrl

    def get_actuator_activations(self, recursive=True) -> np.ndarray:
        """
        Gets the activation values of the actuators in the entity.
        :param recursive: If `True`, the actuator activation values of all attached entities are also retrieved.
        :return: An array containing the activation values of the actuators in the entity.
        """
        return self.__get_actuators_binding(recursive).act

    def get_actuator_controls(self, recursive=True) -> np.ndarray:
        """
        Gets the control values of the actuators in the entity.
        :param recursive: If `True`, the actuator control values of all attached entities are also retrieved.
        :return: An array containing the control values of the actuators in the entity.
        """
        return self.__get_actuators_binding(recursive).ctrl

    def set_state(self, position=None, velocity=None, act=None, ctrl=None, recursive=True) -> None:
        """
        Sets the state of the entity, including joint positions and velocities, and actuator activations and controls.
        :param position: The joint positions to set. If None, the positions are not changed.
        :param velocity: The joint velocities to set. If None, the velocities are not changed.
        :param act: The actuator activations to set. If None, the activations are not changed.
        :param ctrl: The actuator controls to set. If None, the controls are not changed.
        :param recursive: If `True`, the set the state of all attached entities.
        :return:
        """
        self.configure_joints(position, velocity, recursive)
        self.configure_actuators(act, ctrl, recursive)

    # ===================================== #
    # ========== binding helpers ========== #
    # ===================================== #

    def __get_joints_binding(self, recursive=True):
        joints = self.mjcf_element.find_all('joint', exclude_attachments=not recursive)
        return self._physics.bind(joints)

    def __get_actuators_binding(self, recursive=True):
        actuators = self.mjcf_element.root.find_all('actuator', exclude_attachments=not recursive)
        return self._physics.bind(actuators)
