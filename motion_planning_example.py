import numpy as np
from n_table_blocks_world.n_table_blocks_world import NTableBlocksWorld
from motion_planning.motion_planner import NTableBlocksWorldMotionPlanner
from klampt import vis


facing_down_R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]

colors = {"red": [1, 0, 0, 1],
          "yellow": [1, 1, 0, 1],
          "purple": [1, 0, 1, 1],
          "cyan": [0, 1, 1, 1],}


def update_blocks_position(planner, blocks_positions_dict):
    for name, pos in blocks_positions_dict.items():
        planner.move_block(name, pos)


if __name__ == '__main__':
    mp_vis = True
    tol = 0.05
    vel = 0.1

    # vis.init("HTML")

    env = NTableBlocksWorld()
    planner = NTableBlocksWorldMotionPlanner(env)

    state = env.reset()
    blocks = state['object_positions']
    for name, pos in blocks.items():
        planner.add_block(name, pos)

    if mp_vis:
        planner.visualize()

    red_box_pos = env.get_object_pos('block 1 red')
    yellow_box_pos = env.get_object_pos('block 2 yellow')
    purple_box_pos = env.get_object_pos('block 3 cyan')

    robot_joint_pos = state['robot_joint_pos']
    # run simulation for few steps:
    for i in range(10):
        env.step(robot_joint_pos)

    above_red_box_pos = red_box_pos + np.array([0, 0, 0.1])
    path = planner.plan_from_config_to_pose(robot_joint_pos, above_red_box_pos, facing_down_R)
    if mp_vis:
        planner.show_path_vis(path)
    for j in path[1:]:
        state, success = env.move_to(j, tolerance=tol, end_vel=vel)
    env.set_gripper(closed=True)

    update_blocks_position(planner, state['object_positions'])

    # red_box_grasp_config = planner.ik_solve(red_box_pos, facing_down_R, env.robot_joint_pos)
    # env.move_to(red_box_grasp_config, tolerance=tol, end_vel=vel)

    above_purple_box_pos = purple_box_pos + np.array([0, 0, 0.1])
    joint_state = env.robot_joint_pos
    path = planner.plan_from_config_to_pose(joint_state, above_purple_box_pos, facing_down_R)
    for j in path:
        state, success = env.move_to(j, tolerance=tol, end_vel=vel)
    env.set_gripper(closed=False)
    for i in range(10):
        env.step(env.robot_joint_pos)

    update_blocks_position(planner, state['object_positions'])

    above_yellow_box_pos = yellow_box_pos + np.array([0, 0, 0.1])
    joint_state = env.robot_joint_pos
    path = planner.plan_from_config_to_pose(joint_state, above_yellow_box_pos, facing_down_R)
    for j in path:
        state, success = env.move_to(j, tolerance=tol, end_vel=vel)
    env.set_gripper(closed=True)

    red_box_new_pos = env.get_object_pos('block 1 red')
    above_tower_pos = red_box_new_pos + np.array([0, 0, 0.1])
    joint_state = env.robot_joint_pos
    path = planner.plan_from_config_to_pose(joint_state, above_tower_pos, facing_down_R)
    for j in path:
        state, success = env.move_to(j, tolerance=tol, end_vel=vel)
    env.set_gripper(closed=False)
    for i in range(10):
        env.step(env.robot_joint_pos)

    # move robot to home position
    home_js = np.array([0, -1.57, 0, 0, 0, 0])
    for i in range(1000):
        env.step(home_js)





