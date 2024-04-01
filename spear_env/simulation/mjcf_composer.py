from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from dm_control import mjcf

from .mjcf_utils import load_mjcf, attach_addon_collection, attach_addon, attach_mjcf
from ..episode.specs.robot_spec import RobotSpec, AttachmentSpec
from ..episode.specs.scene_spec import SceneSpec, ObjectSpec


class MJCFComposer:
    """
    A composition tool for MJCF assets.
    The tool uses PyMJCF (see https://github.com/deepmind/dm_control/blob/main/dm_control/mjcf/README.md) to attach
    desired assets to a base scene. All assets are specified as `AssetSpec` objects (see
    `spear_env.episode.specs.AssetSpec`) from which the asset is loaded and attached accordingly. The composer object
    stitches the assets appropriately, cleverly handling issues like name ambiguity and keyframe data.

    public fields:
        - mjcf_model:
            An MJCF root element (see `dm_control.mjcf.element.RootElement`) that represents the MJCF of the entire
            scene, including all attached assets.

    public methods:
        - reset:
            reset the composer to an empty model with no additional assets.
        - set_base_scene:
            set the base MJCF according to a specific scene specification. This will trigger `reset`.
        - attach_object / attach_robot
            attach a robot or an object to the scene according to the provided specification.
        - get_object / get_robot
            Retrieves models of the loaded assets that were attached to the scene by the specification object with
            which the `attach_object` and `attach_robot` methods were called.
        - get_mounted_robot:
            When a robot specification contains a mount specification, the robot is attached to the mount model. Using
            the same robot specification object, this method retrieves the robot's mount model with containing the
            attached robot.
        - get_attachment:
            When a robot specification contains attachment specifications, they are attached to the robot model. This
            method retrieves the model of the attachment with the attachment specification object with which it was
            generated.
        - swap_spec_ids:
            Swaps the ids of objects and robots in the composer with the ones provided. This is used to update the
            lookup tables when the specification changes without requiring a new model to be composed.
    """
    def __init__(self):
        """Initializes the root scene and addon lookup tables (objects, robots, attachments, mounted_robots)"""
        self._base_model: mjcf.RootElement = mjcf.RootElement()
        self._objects: dict[ObjectSpec, mjcf.RootElement] = {}
        self._robots: dict[RobotSpec, mjcf.RootElement] = {}
        self._attachments: dict[RobotSpec, dict[AttachmentSpec, mjcf.RootElement]] = {}
        self._mounted_robots: dict[RobotSpec, mjcf.RootElement] = {}

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """
        The complete MJCF scene composed thus far.
        :return: An `mjcf.RootElement` object.
        """
        return self._base_model

    def reset(self):
        """reset the composer to an empty model with no additional assets."""
        self._base_model = mjcf.RootElement()
        self._objects = {}
        self._robots = {}
        self._attachments = {}
        self._mounted_robots = {}

    def set_base_scene(self, base_scene: SceneSpec):
        """
        set the base MJCF according to a specific scene specification. This will trigger `reset`.
        :param base_scene: A scene specification object.
        """
        # reset composer
        self.reset()

        # load scene resource
        self._base_model = load_mjcf(base_scene.resource)

        # add objects to scene
        mjcf_objects = attach_addon_collection(base_scene.objects, self._base_model.worldbody)

        # update stores
        self._objects = {spec: mjcf_obj for spec, mjcf_obj in zip(base_scene.objects, mjcf_objects)}

    def attach_object(self, spec: ObjectSpec):
        """
        attach an object to the scene according to the provided specification.
        :param spec: An object specification object
        """
        # add object to scene and update stores
        self._objects[spec] = attach_addon(spec, self._base_model.worldbody)

    def attach_robot(self, spec: RobotSpec):
        """
        attach a robot to the scene according to the provided specification.
        :param spec: A robot specification object
        """
        mjcf_robot = load_mjcf(spec.resource)
        if spec.mount is None:  # no mount specified
            mjcf_mounted_robot = self._attach_robot_without_mount(mjcf_robot, spec)
        else:  # mount specified
            mjcf_mounted_robot = self._attach_robot_with_mount(mjcf_robot, spec)

        # add attachments
        mjcf_attachments = attach_addon_collection(spec.attachments, mjcf_robot.worldbody)

        # update stores
        self._robots[spec] = mjcf_robot
        self._mounted_robots[spec] = mjcf_mounted_robot
        self._attachments[spec] = {spec_att: mjcf_att
                                      for spec_att, mjcf_att in zip(spec.attachments, mjcf_attachments)}

    def get_object(self, spec: ObjectSpec, as_attachment_element=True) -> mjcf.Element:
        """
        Retrieves a model of a loaded asset that was attached to the scene by a specification object with which the
        `attach_object` method was called.
        :param spec: An object specification object
        :param as_attachment_element: If `False`, the returned element is the root element loaded from the asset MJCF
                                      file. If `True`, instead return the element in the base model that represents
                                      this attached asset.
        :return: An MJCF element that was generated by the given specification.
        """
        return self.__maybe_convert_to_attachment(self._objects[spec], as_attachment_element)

    def get_robot(self, spec: RobotSpec, as_attachment_element=True) -> mjcf.Element:
        """
        Retrieves a model of a loaded asset that was attached to the scene by a specification object with which the
        `attach_robot` method was called.
        :param spec: A robot specification object
        :param as_attachment_element: If `False`, the returned element is the root element loaded from the asset MJCF
                                      file. If `True`, instead return the element in the base model that represents
                                      this attached asset.
        :return: An MJCF element that was generated by the given specification.
        """
        return self.__maybe_convert_to_attachment(self._robots[spec], as_attachment_element)

    def get_mounted_robot(self, spec: RobotSpec, as_attachment_element=True) -> mjcf.Element:
        """
        Retrieves a model of a specified mount for a robot asset that was attached to the scene using `attach_robot`
        method. The returned model is the one of the mount with the robot attached to it.
        :param spec: A robot specification object
        :param as_attachment_element: If `False`, the returned element is the root element loaded from the asset MJCF
                                      file. If `True`, instead return the element in the base model that represents
                                      this attached asset.
        :return: An MJCF element for the mount (with attached robot) that was generated by the given specification.
        """
        return self.__maybe_convert_to_attachment(self._mounted_robots[spec], as_attachment_element)

    def get_attachment(self, attachment_spec: AttachmentSpec, robot_spec: Optional[RobotSpec] = None,
                       as_attachment_element=True) -> mjcf.Element:
        """
        Retrieves a model of a specified mount for a robot asset that was attached to the scene using `attach_robot`
        method. The returned model is the one of the mount with the robot attached to it.
        :param attachment_spec: An attachment specification object
        :param robot_spec: An optional robot specification object for the parent robot (for faster search).
        :param as_attachment_element: If `False`, the returned element is the root element loaded from the asset MJCF
                                      file. If `True`, instead return the element in the base model that represents
                                      this attached asset.
        :return: An MJCF element that was generated by the given attachment specification.
        """
        if robot_spec is None:
            for robot_spec, attachment_dict in self._attachments.items():
                if attachment_spec in attachment_dict:
                    return self.__maybe_convert_to_attachment(attachment_dict[attachment_spec], as_attachment_element)
            raise KeyError(f"Attachment {attachment_spec} not found in any robot.")
        else:
            return self.__maybe_convert_to_attachment(self._attachments[robot_spec][attachment_spec], as_attachment_element)

    def swap_spec_ids(self, objs_in, objs_out, robots_in, robots_out):
        """
        Swaps the ids of objects and robots in the composer with the ones provided.
        """
        self._swap_objects(objs_in, objs_out)
        self._swap_robots(robots_in, robots_out)

    def extract_keyframes(self) -> dict[str, list[mjcf.RootElement]]:
        """
        Extracts keyframe data from the base model. The keyframes are later removed from the model to avoid mismatched
        keyframe data for mjcf attachments.
        :return: A dictionary mapping keyframe names to their data.
        """
        keyframes = OrderedDict()

        for key in self._base_model.keyframe.key:
            # set a name for the keyframe if it doesn't have one
            key_name = key.name if key.name else key.full_identifier

            # add keyframe data to the dictionary
            keyframes.setdefault(key_name, []).append((
                key.namescope.mjcf_model,  # the model to which the keyframe applies
                key.qpos,  # the joint position data
                key.qvel,  # the joint velocity data
                key.act,  # the actuator activation data
                key.ctrl  # the actuator control data
            ))

            # remove keyframe from the model
            key.remove()

        return keyframes

    def _attach_robot_with_mount(self, mjcf_robot, spec):
        # load mount MJCF asset file
        mjcf_mounted_robot = load_mjcf(spec.mount.resource)

        # we expect the mount asset to have a single body that is the root body of the mount.
        # we attach robot to the root body (the base) of the mount to account for offset to mount base.
        # this can be overriden by providing `base_pos` and `base_rot` or an existing `site` in the mount spec.
        # this ensures that the robot is attached to the mount at the correct location relative to the mount's base.
        # this positioning can be adjusted by changing the mount's `base_pos` attribute.
        mount_root = mjcf_mounted_robot.worldbody.find_all('body', immediate_children_only=True)[0]
        attach_mjcf(mjcf_robot, mount_root, spec.mount)

        # attach mount to the base site added to the scene
        attach_mjcf(mjcf_mounted_robot, self._base_model.worldbody, spec)
        return mjcf_mounted_robot

    def _attach_robot_without_mount(self, mjcf_robot, spec):
        # attach robot to the base site added to the scene
        attach_mjcf(mjcf_robot, self._base_model.worldbody, spec)

        # set mount attributes to None (for consistency with episode)
        mjcf_mounted_robot = mjcf_robot  # robot base is its own mount
        return mjcf_mounted_robot

    @staticmethod
    def __maybe_convert_to_attachment(item: mjcf.RootElement, convert):
        if convert:
            return mjcf.get_attachment_frame(item)
        return item

    def _swap_robots(self, robots_in, robots_out):
        for robot_in, robot_out in zip(robots_in, robots_out):
            self._robots[robot_in] = self._robots.pop(robot_out)
            self._mounted_robots[robot_in] = self._mounted_robots.pop(robot_out)
            self._swap_attachments(robot_in, robot_out)
            self._attachments[robot_in] = self._attachments.pop(robot_out)

    def _swap_attachments(self, robot_in, robot_out):
        for attachment_in, attachment_out in zip(robot_in.attachments, robot_out.attachments):
            self._attachments[robot_out][attachment_in] = self._attachments[robot_out].pop(attachment_out)

    def _swap_objects(self, objs_in, objs_out):
        for obj_in, obj_out in zip(objs_in, objs_out):
            self._objects[obj_in] = self._objects.pop(obj_out)
