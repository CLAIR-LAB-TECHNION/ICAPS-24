from motion_planner import TableWorldMotionPlanner
import time


if __name__ == '__main__':


    planner = TableWorldMotionPlanner()
    start_config = [0, 0, 0, 0, 0, 0]

    goal_pos = [-0.19144999999599827, 0.4172500000009377, 0.97135]
    goal_R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    goal_config = planner.ik_solve_6d(goal_pos, goal_R)
    if goal_config is None:
        print("no ik solution for goal")
        exit(0)

    path = planner.plan_from_start_to_goal_config(start_config, goal_config)

    print("path:")
    for p in path:
        print(p)

    planner.open_vis()
    planner.move_to_config_vis(goal_config)

    planner.vis_spin(5000)
    # for p in path:
    #     planner.move_to_config_vis(p)
    #     planner.vis_spin(1)

