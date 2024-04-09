import time
from typing import Optional

from .base_renderer import BaseRenderer

import mujoco
import mujoco.viewer as mj_viewer


class WindowRenderer(BaseRenderer):
    def _initialize(self, render_fps: int = 60, scene_flags: Optional[mujoco.mjtRndFlag] = None,
                    vis_flags: Optional[mujoco.mjtRndFlag] = None) -> None:
        self.render_spf = 1 / render_fps

        # initialize viewer
        self.viewer = mj_viewer.launch_passive(self._model, self._data)

        with self.viewer.lock():
            # set render flags
            for flag in (scene_flags or []):
                self.scene.flags[flag] = True

            # set vis flags
            for flag in (vis_flags or []):
                self.options.flags[flag] = True

            # set scene maxgeoms for cluttered scenes
            self.scene.maxgeom = self._model.ngeom * 2

            # set camera
            self.camera.fixedcamid = self._camera_id
            if self._camera_id != -1:
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

        self.prev_render_time = 0

    @property
    def scene(self) -> mujoco.MjvScene:
        return self.viewer.user_scn

    @property
    def camera(self) -> mujoco.MjvCamera:
        return self.viewer.cam

    @property
    def options(self) -> mujoco.MjvOption:
        return self.viewer.opt

    @property
    def perturbations(self) -> mujoco.MjvPerturb:
        return self.viewer.perturb

    def render(self) -> None:
        # sleep to maintain FPS
        cur_time = time.time()
        time_to_next_frame = self.render_spf - (cur_time - self.prev_render_time)
        if time_to_next_frame > 0:
            time.sleep(time_to_next_frame)

        # update render window
        self.viewer.sync()

        # update time
        self.prev_render_time = time.time()

    def close(self) -> None:
        self.viewer.close()
