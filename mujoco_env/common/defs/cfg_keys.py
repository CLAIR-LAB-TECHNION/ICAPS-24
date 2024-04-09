##################
# top level keys #
##################

GLOBALS = 'globals'
EPISODES = 'episodes'

####################
# the resource key #
####################
# this is a key used at the top level of all specifications that reference an external asset to be attached to the
# scene, including the scene itself, the robot, attachments, mounts, and objects.

RESOURCE = 'resource'

#####################
# episode spec keys #
#####################

SCENE = 'scene'
ROBOT = 'robot'
TASK = 'task'

###################
# scene spec keys #
###################

SCENE_OBJECTS = 'objects'
SCENE_RENDER_CAMERA = 'render_camera'
SCENE_INIT_KEYFRAME = 'init_keyframe'

###################
# robot spec keys #
###################

ROBOT_CAMERAS = 'cameras'
ROBOT_ATTACHMENTS = 'attachments'
ROBOT_MOUNT = 'mount'
ROBOT_PRIVILEGED_INFO = 'privileged_info'

# duplicates of the ADDON_* keys
ROBOT_SITE = 'site'
ROBOT_BASE_POS = 'base_pos'
ROBOT_BASE_ROT = 'base_rot'
ROBOT_BASE_JOINTS = 'base_joints'
ROBOT_INIT_QPOS = 'init_qpos'
ROBOT_INIT_QVEL = 'init_qvel'

##################
# task spec keys #
##################

TASK_CLASS = 'cls'
TASK_PARAMS = 'params'

########################################################################
# addon spec keys (objects, attachments, mounts, and the robot itself) #
########################################################################

ADDON_SITE = 'site'
ADDON_BASE_POS = 'base_pos'
ADDON_BASE_ROT = 'base_rot'
ADDON_BASE_JOINTS = 'base_joints'

# for addons with controllable init state (objects and robots)
ADDON_INIT_QPOS = 'init_qpos'
ADDON_INIT_QVEL = 'init_qvel'

##################################
# joints and actuators spec keys #
##################################

JOINT_TYPE = 'type'
JOINT_ATTRS = 'attrs'
JOINT_ACTUATORS = 'actuators'
ACTUATOR_TYPE = 'type'
ACTUATOR_ATTRS = 'attrs'
