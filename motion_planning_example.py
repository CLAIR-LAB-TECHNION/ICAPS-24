import numpy as np
from table_world import TableWorld
from motion_planner import TableWorldMotionPlanner


facing_down_R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]

if __name__ == '__main__':
    mp_vis = False

    env = TableWorld()
    planner = TableWorldMotionPlanner(env)

    robot_joint_pos = env.reset()
    # run simulation for few steps:
    for i in range(10):
        env.step(robot_joint_pos)

    red_box_pos = env.get_object_pos('red_box') + np.array([0, 0, 0.015])  # TODO + box size
    above_red_box_pos = red_box_pos + np.array([0, 0, 0.1])
    path = planner.plan_from_config_to_pose(robot_joint_pos, above_red_box_pos, facing_down_R)

    for j in path:
        env.move_to(j, tolerance=0.05)

    red_box_grasp_config = planner.ik_solve(red_box_pos, facing_down_R, env.robot_joint_pos)
    env.move_to(red_box_grasp_config, tolerance=0.05)
    for i in range(5):
        env.set_gripper(closed=True)

    env.max_joint_velocities = 0.1

    purple_box_pos = env.get_object_pos('purple box')
    above_purple_box_pos = purple_box_pos + np.array([0, 0, 0.1])
    joint_state = env.robot_joint_pos
    path = planner.plan_from_config_to_pose(joint_state, above_purple_box_pos, facing_down_R)
    for j in path:
        env.move_to(j, tolerance=0.05)

    env.set_gripper(closed=False)

    js = env.robot_joint_pos
    for i in range(1000):
        env.step(js)





