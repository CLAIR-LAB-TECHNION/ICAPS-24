import numpy as np
from mujoco_env.tasks.null_task import NullTask


block_size = [0.1, 0.1, 0.04]
# blocks are configured at spear_env/assets/scenes/3tableblocksworld/scene.xml
# Right now this is a constant and can't be changed from here
# Note that sizes in the xml file are half, because boxes are defined by their center and half size

robot_height = 0.903 + 0.163 - 0.089159
# 0.903 is the height of the robot mount, 0.163 is the height of the shift of shoulder link in mujoco,
# 0.089159 is the height of shoulder link in urdf for klampt
mount_top_base = robot_height - 0.01  # avoid collision between robot base and mount

table_size = [0.6, 0.6, 0.01]
table_left_pos = [0.0, -0.6, 0.7]
table_right_pos = [0.0, 0.6, 0.7]
table_front_pos = [0.6, 0.0, 0.7]

env_cfg = dict(
    scene=dict(
        resource='3tableblocksworld',
        render_camera='top-right',
        # renderer_cfg={"width": 320, "height": 240},
    ),
    robot=dict(
        resource='ur5e',
        mount='rethink_stationary',
        privileged_info=True,
        attachments=['adhesive_gripper'],
    ),
    task=NullTask,
)
# for spear env

INIT_CONFIG = np.array([0, -1.57, 0, 0, 0, 0])
INIT_MAX_VELOCITY = np.array([2, 2, 2, 2, 2, 2])

# relative position of grasped object from end effector
grasp_offset = 0.02


frame_skip = 5


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