from motion_planner import TableWorldMotionPlanner
import time
import numpy as np
import spear_env
from spear_env.tasks.null_task import NullTask


env_cfg = dict(
    scene=dict(
        resource='tableworld',
        render_camera='top-right'
    ),
    robot=dict(
        resource='ur5e',
        mount='rethink_stationary',
        privileged_info=True,
        attachments=['adhesive_gripper'],
    ),
    task=NullTask,
)

purpule_box_pos = [-0.2, 0.5, 0.72]
red_box_pos = [0, -0.8, 0.72]

facing_down_R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]


def move_to_config(env, config, tolerance=0.05, grasp=False, sleep_time=0.02):
    action = np.concatenate((config, [int(grasp)]))  # add gripper action
    obs, r, term, trunc, info = env.step(action)
    while np.linalg.norm(np.array(obs['robot_state'][:6]) - np.array(config)) > tolerance:
        time.sleep(0.02)
        obs, r, term, trunc, info = env.step(action)
        env.render()

    return obs, r, term, trunc, info


def move_path(env, path, tolerance=0.05, grasp=True, sleep_time=0.02):
    assert len(path) > 0
    for config in path:
        obs, r, term, trunc, info = move_to_config(env, config, tolerance, grasp, sleep_time)
    return obs, r, term, trunc, info


def get_object_pos(info, obj_name):
    return info['priveleged']['model'].body(obj_name).pos


def pritn_objects_name(info):
    for i in range(info['priveleged']['model'].nbody):
        print(info['priveleged']['model'].body(i).name)


def plan_and_move_to_pose(env, planner, last_obs, goal_pos, goal_R, max_time=30, tolerance=0.05, grasp=False,
                          sleep_time=0.02, mp_vis=False):
    start_config = last_obs['robot_state'][:6]
    path = planner.plan_from_config_to_pose(start_config, goal_pos, goal_R, max_time)
    if mp_vis:
        planner.show_path_vis(path)
    obs, r, term, trunc, info = move_path(env, path, tolerance, grasp, sleep_time)
    return obs, r, term, trunc, info


if __name__ == '__main__':
    mp_vis = True

    env = spear_env.from_cfg(cfg=env_cfg, render_mode="human", frame_skip=5)

    planner = TableWorldMotionPlanner(eps=0.02)
    obs, info = env.reset()

    # here:
    # TODO: wrapper and use env.agent.entity.set_state(velocity=<clipped_velocity>)

    obs, _, _, _, _, = move_to_config(env, [0, -1.57, 0, 0, 0, 0])
    if mp_vis:
        planner.open_vis()

    above_red_box_pos = red_box_pos.copy()
    above_red_box_pos[2] += 0.1
    obs, _, _, _, _ = plan_and_move_to_pose(env, planner, obs, above_red_box_pos, facing_down_R, mp_vis=mp_vis)
    red_box_grasp_config = planner.ik_solve(red_box_pos, facing_down_R, obs['robot_state'][:6],)
    obs, _, _, _, _ = move_to_config(env, red_box_grasp_config)
    action = np.concatenate((obs["robot_state"][:6], [1]))  # close gripper
    for i in range(20):
        obs, r, term, trunc, info = env.step(action)

    above_purple_box_pos = purpule_box_pos.copy()
    above_purple_box_pos[2] += 0.1
    obs, _, _, _, _ = plan_and_move_to_pose(env, planner, obs, above_purple_box_pos, facing_down_R,  grasp=True, mp_vis=mp_vis)

    action = np.concatenate((obs["robot_state"][:6], [0]))  # open gripper
    for i in range(10):
        obs, r, term, trunc, info = env.step(action)

    while True:
        move_to_config(env, obs["robot_state"][:6])




    # planner.open_vis()
    # planner.vis_spin(2)
    # for p in path_interpolated:
    #     planner.move_to_config_vis(p)
    #     planner.vis_spin(0.1)
    #
    #
    # planner.vis_spin(5000)


