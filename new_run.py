import numpy as np

import spear_env
from spear_env.tasks.null_task import NullTask

from spear_env.episode import *
from spear_env.common.defs import JointType, ActuatorType, cfg_keys


cfg = dict(
    scene=dict(
        resource='tableworld',
        render_camera='top-right'
    ),
    robot=dict(
        resource='ur5e',
        mount='rethink_stationary',
        # attachments=['adhesive_gripper'],
    ),
    task=NullTask,
)

env = spear_env.from_cfg(cfg=cfg, render_mode="human", frame_skip=5)

N_EPISODES = 1
N_STEPS = 20000

try:
    for _ in range(N_EPISODES):
        obs, info = env.reset()
        env.render()
        # frames.append(obs['camera_fetch/tracker_480X640_rgb'])
        done = False
        i = 0
        while not done and i < N_STEPS:
            i += 1
            action = env.action_space.sample()
            action = np.zeros_like(action)
            action = [-0.5, 0.1, 0.2, -0.5, -0.5, -0.5]

            obs, r, term, trunc, info = env.step(action)
            if i == 200 or i==1:
                print(obs)
            done = term or trunc
            # frames.append(obs['camera_fetch/tracker_480X640_rgb'])
            env.render()
except KeyboardInterrupt:
    pass

# from moviepy.editor import ImageSequenceClip
# clip = ImageSequenceClip(frames, fps=env.metadata['render_fps'])
# clip.write_videofile('test.mp4')

env.close()
