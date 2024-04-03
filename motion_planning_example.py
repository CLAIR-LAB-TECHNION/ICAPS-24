import numpy as np
from block_world import BlockWorld
from motion_planner import BlockWorldMotionPlanner


facing_down_R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]

if __name__ == '__main__':
    mp_vis = False
    tol = 0.05

    env = BlockWorld()
    planner = BlockWorldMotionPlanner(env)

    robot_joint_pos = env.reset()
    # run simulation for few steps:
    for i in range(10):
        env.step(robot_joint_pos)

    red_box_pos = env.get_object_pos('red_box') + np.array([0, 0, 0.02])  # TODO + box size
    above_red_box_pos = red_box_pos + np.array([0, 0, 0.1])
    path = planner.plan_from_config_to_pose(robot_joint_pos, above_red_box_pos, facing_down_R)

    for j in path:
        env.move_to(j, tolerance=tol)

    red_box_grasp_config = planner.ik_solve(red_box_pos, facing_down_R, env.robot_joint_pos)
    env.move_to(red_box_grasp_config, tolerance=tol)
    for i in range(10):
        env.set_gripper(closed=True)

    env.max_joint_velocities = 0.1

    purple_box_pos = env.get_object_pos('purple box')
    above_purple_box_pos = purple_box_pos + np.array([0, 0, 0.1])
    joint_state = env.robot_joint_pos
    path = planner.plan_from_config_to_pose(joint_state, above_purple_box_pos, facing_down_R)
    for j in path:
        env.move_to(j, tolerance=tol)

    env.set_gripper(closed=False)

    js = env.robot_joint_pos
    for i in range(1000):
        env.step(js)





