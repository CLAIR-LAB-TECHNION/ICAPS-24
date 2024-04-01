import os
import warnings
import tempfile
from typing import Iterable

import mujoco
from dm_control import mjcf
from dm_control.mjcf.element import _AttachableElement as Attacchable

from ..common import FilePath
from ..episode.specs.addon_spec import AddonSpec
from ..episode.specs.joint_spec import JointSpec, ActuatorSpec

__all__ = [
    'load_mjcf',
    'physics_from_mjcf_model',
    'attach_addon_collection',
    'attach_addon',
    'attach_mjcf',
    'joint_element_to_spec',
    'get_element_from_name_and_tag'
]


def load_mjcf(model_path: FilePath):
    """
    Loads an MJCF model from a given path. Supports URDF files by converting them to MJCF.
    :param model_path: The path to the MJCF model.
    :return: An MJCF root element for the loaded model.
    """
    # attempt to load as MJCF with PyMJCF
    try:
        return mjcf.from_path(model_path)
    except (ValueError, KeyError, FileNotFoundError):
        pass  # not supported by PyMJCF

    # attempt to load with MuJoCo and save as a combined MJCF with all include files already loaded.
    # This is done to support URDF files and other PyMJCF issues with include files.

    # extract model path components
    model_dir, model_fname = os.path.split(model_path)
    model_name, model_ext = os.path.splitext(model_fname)

    mjcf_model_name = model_name + '__mjcf_tmp' + model_ext
    mjcf_model_path = os.path.join(model_dir, mjcf_model_name)

    # load as model object and save as full xml
    model = mujoco.MjModel.from_xml_path(str(model_path))
    mujoco.mj_saveLastXML(mjcf_model_path, model)

    # remove default with class "main" from newly saved mjcf file
    with open(mjcf_model_path, 'r') as f:
        content = f.read()
    content = content.replace('default class="main"', 'default')
    with open(mjcf_model_path, 'w') as f:
        f.write(content)

    # load temporary file to memory and delete
    mjcf_model = mjcf.from_path(mjcf_model_path)
    os.remove(mjcf_model_path)

    return mjcf_model


def physics_from_mjcf_model(mjcf_model: mjcf.RootElement) -> mjcf.Physics:
    """
    Creates a physics model and state from an MJCF model.
    :param mjcf_model: The MJCF root model to be loaded as a physics object
    :return: A `dm_control.mjcf.Physics` object
    """
    try:
        return mjcf.Physics.from_mjcf_model(mjcf_model)
    except ValueError:
        # a workaround for an issue with loading physics from PyMJCF models.
        # issue details: https://github.com/google-deepmind/mujoco/issues/1054
        # workaround: export with assets to temporary file and load physics from file
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_filename = f'{mjcf_model.model}.xml'
            mjcf.export_with_assets(mjcf_model, tmp_dir, tmp_filename)
            return mjcf.Physics.from_xml_path(f'{tmp_dir}/{tmp_filename}')


def attach_addon_collection(addon_list: Iterable[AddonSpec], attachable: Attacchable):
    """
    attaches a list of addons to a given attachable MJCF element.
    :param addon_list: The list of specifications for addons to attach.
    :param attachable: The attachable element to which the addons are attached.
    :return: A list of MJCF root elements for the loaded assets of the provided addons.
    """
    return [
        attach_addon(addon, attachable)
        for addon in addon_list
    ]


def attach_addon(addon: AddonSpec, attachable: Attacchable):
    """
    attaches an addon to a given attachable MJCF element.
    :param addon The specification for the addon to attach.
    :param attachable: The attachable element to which the addon is attached.
    :return: An MJCF root element for the loaded asset of the provided addon.
    """
    # load addon MJCF asset files
    addon_mjcf = load_mjcf(addon.resource)

    # attach addon to attachable parent
    attach_mjcf(addon_mjcf, attachable, addon)

    return addon_mjcf


def attach_mjcf(src_mjcf, dest_element, addon: AddonSpec):
    """
    Attach a source MJCF element to a destination MJCF element.
    :param src_mjcf: The source MJCF element.
    :param dest_element: The destination MJCF element.
    :param addon: The addon specification that generated `src_mjcf`.
    :return: The attachment site (mjcf element) to which the element was attached in the destination.
    """
    # set base joints
    base_joints = __select_model_base_joints(src_mjcf, addon.base_joints)

    # attach source mjcf to destination mjcf with base joints in the attachment frame wrapper body
    site = __get_attachment_site(dest_element, addon.site)
    __set_site_pose(site, addon.base_pos, addon.base_rot)
    __attach_to_site(site, src_mjcf, base_joints)

    return site


def joint_element_to_spec(joint_element: mjcf.Element):
    """
    convert an MJCF element representing a joint to a joint specification object
    :param joint_element: The MJCF joint element
    :return: A matching specification for the joint.
    """
    jnt_attr = joint_element.get_attributes()
    actuators = []
    for a in joint_element.root.find_all('actuator'):
        if a.joint == joint_element:
            act_attrs = a.get_attributes()
            act_attrs.pop('joint', None)  # remove joint attribute and add parent jointspec
            actuators.append(ActuatorSpec(type=a.tag, attrs=act_attrs))
            a.remove()
    return JointSpec(type=jnt_attr.pop('type', 'free'), actuators=actuators, attrs=jnt_attr)


def get_element_from_name_and_tag(name: str, tag: str, mjcf_model: mjcf.RootElement):
    """
    find an element in an MJCF model with recursive search within attached models. The search is performed as follows:
        - look for items that match the name perfectly according to the PyMJCF naming convention.
        - look for items that match the name perfectly within any of the attachments (in order of attachment).
        - look for attachment frames (body tag only) that carry the same model name as the search name.
        - look for items that match the name perfectly, considering hidden elements due to `find` method restrictions.
    :param name: the name of the element to find.
    :param tag: the XML tag of the element to find.
    :param mjcf_model: the root MJCF element in which to begin the search.
    :return: The desired MJCF element.
    :raises: ValueError if the desired element is not found
    """
    # try to find element by name
    mjcf_element = mjcf_model.find(tag, name)
    if mjcf_element is not None:
        return mjcf_element

    # try to find element by name in attachments
    for attachment in mjcf_model._attachments:
        try:
            mjcf_element = get_element_from_name_and_tag(name, tag, attachment.mjcf_model)
            return mjcf_element  # return if found without error
        except ValueError:
            pass

    if tag == 'body':
        # try to find an attachment with the same model name
        for attachment in mjcf_model._attachments:
            if attachment.mjcf_model.model == name:
                return mjcf.get_attachment_frame(attachment.mjcf_model)

    # try to find element by full identifier. this can reveal attachment frames with attachment delimiters "/" that
    # cannot be found using the `find` method
    matching_tags = [b for b in mjcf_model.worldbody.find_all(tag) if b.full_identifier == name]
    if len(matching_tags) > 1:
        raise ValueError(f'Multiple instances of <{tag}> tag with name "{name}" found in model')
    elif len(matching_tags) == 1:
        return matching_tags[0]

    # element not found
    raise ValueError(f'<{tag}> tag with name "{name}" not found in model {mjcf_model.model}')


#######################################
# Handle base joints helper functions #
#######################################


def __select_model_base_joints(model_mjcf, addon_base_joints):
    root_bodies = model_mjcf.worldbody.find_all('body', immediate_children_only=True)

    if len(root_bodies) != 1:
        return addon_base_joints  # no bodies or multiple bodies are attached as is with the given base joints

    freejoint = mjcf.get_freejoint(root_bodies[0])
    if freejoint:
        # cannot have freejoint inside attachment. moving to attachment frame
        freejoint.remove()

    if freejoint and addon_base_joints:
        # internal freejoint must be overriden. Attachments are wrapped in a body tag so internal freejoints will not be
        # in a direct child of worldbody. We mmove the freejoint to the wrapper body but warn the user.
        warnings.warn(f'the freejoint included in model "{model_mjcf.model}" was overriden by provided base joints: '
                      f'{addon_base_joints}')
        __remove_unlinked_actuators(model_mjcf)

    if addon_base_joints:
        return addon_base_joints
    elif freejoint:
        return [joint_element_to_spec(freejoint)]
    else:
        return []


def __remove_unlinked_actuators(model_mjcf):
    actuator_list = model_mjcf.root.find_all('actuator')

    for a in actuator_list:
        if a.joint is None:
            a.remove()


def __dummy_base_joints_body(dest_element, base_joints):
    body = dest_element.worldbody.add('body')
    for joint_spec in base_joints:
        body.add('joint', type=joint_spec.type.value, **joint_spec.attrs)
    body.add('geom', size='1')  # add dummy geom to add mass to the body and prevent errors

    return body


###############################
# Attachment helper functions #
###############################


def __get_attachment_site(dest_element, site_name=None):
    if site_name is None:
        site = dest_element.add('site', size=[1e-6] * 3)  # create dummy site
    else:
        site = get_element_from_name_and_tag(site_name, 'site', dest_element.root)
        if site is None:  # attachment site MUST exist for attachment placement
            raise ValueError(
                f'attachment site "{site_name}" not found in robot'
            )

    return site


def __set_site_pose(site, base_pos, base_rot):
    if base_pos is not None:
        # override given position in site
        site.pos = base_pos

    if base_rot is not None:
        # override given rotation in site
        for rot in ['quat', 'axisangle', 'euler', 'xyaxes', 'zaxis']:
            setattr(site, rot, None)  # remove rotation type from attributes

        # choose rotation type based on number of values given
        if len(base_rot) == 4:  # quaternion (may also be axis-angle, but is ignored in this case)
            site.quat = base_rot
        elif len(base_rot) == 3:  # euler angles (may also be z-axis, but is ignored in this case)
            site.euler = base_rot
        elif len(base_rot) == 6:  # x and y axes
            site.xyaxes = base_rot
        else:  # invalid rotation
            raise ValueError(
                f'invalid rotation "{base_rot}" for attachment. '
                f'expected quaternion (4 values) or euler angles (3 values)'
            )


def __attach_to_site(site, src_mjcf, base_joints=None):
    attachment_frame = site.attach(src_mjcf)
    __add_attachment_base_joints(attachment_frame, base_joints)


def __add_attachment_base_joints(attachment_frame, base_joints=None):
    if base_joints is None:
        return

    for i, joint_spec in enumerate(base_joints):
        joint = attachment_frame.add('joint', type=joint_spec.type.value, **joint_spec.attrs)
        for actuator in joint_spec.actuators:
            attachment_frame.root.actuator.add(actuator.type.value, joint=joint, **actuator.attrs)
