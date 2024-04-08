import numpy as np
from n_table_blocks_world.n_table_blocks_world import NTableBlocksWorld
from motion_planning.motion_planner import NTableBlocksWorldMotionPlanner


facing_down_R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]


if __name__ == '__main__':
    mp_vis = True
    tol = 0.05
    vel = 0.1

    env = NTableBlocksWorld()
    planner = NTableBlocksWorldMotionPlanner(env)
    if mp_vis:
        planner.open_vis()

    red_box_pos = env.get_object_pos('block 10 yellow')
    yellow_box_pos = env.get_object_pos('block 2 yellow')
    purple_box_pos = env.get_object_pos('block 3 cyan')

    state = env.reset()
    robot_joint_pos = state['robot_joint_pos']
    # run simulation for few steps:
    for i in range(10):
        env.step(robot_joint_pos)

    above_red_box_pos = red_box_pos + np.array([0, 0, 0.1])
    path = planner.plan_from_config_to_pose(robot_joint_pos, above_red_box_pos, facing_down_R)
    if mp_vis:
        planner.show_path_vis(path)
    for j in path[1:]:
        env.move_to(j, tolerance=tol, end_vel=vel)
    env.set_gripper(closed=True)

    # red_box_grasp_config = planner.ik_solve(red_box_pos, facing_down_R, env.robot_joint_pos)
    # env.move_to(red_box_grasp_config, tolerance=tol, end_vel=vel)

    above_purple_box_pos = purple_box_pos + np.array([0, 0, 0.1])
    joint_state = env.robot_joint_pos
    path = planner.plan_from_config_to_pose(joint_state, above_purple_box_pos, facing_down_R)
    for j in path:
        env.move_to(j, tolerance=tol, end_vel=vel)
    env.set_gripper(closed=False)
    for i in range(10):
        env.step(env.robot_joint_pos)

    above_yellow_box_pos = yellow_box_pos + np.array([0, 0, 0.1])
    joint_state = env.robot_joint_pos
    path = planner.plan_from_config_to_pose(joint_state, above_yellow_box_pos, facing_down_R)
    for j in path:
        env.move_to(j, tolerance=tol, end_vel=vel)
    env.set_gripper(closed=True)

    red_box_new_pos = env.get_object_pos('block 1 red')
    above_tower_pos = red_box_new_pos + np.array([0, 0, 0.1])
    joint_state = env.robot_joint_pos
    path = planner.plan_from_config_to_pose(joint_state, above_tower_pos, facing_down_R)
    for j in path:
        env.move_to(j, tolerance=tol, end_vel=vel)
    env.set_gripper(closed=False)
    for i in range(10):
        env.step(env.robot_joint_pos)

    # move robot to home position
    home_js = np.array([0, -1.57, 0, 0, 0, 0])
    for i in range(1000):
        env.step(home_js)





