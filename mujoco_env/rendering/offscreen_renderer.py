from typing import Optional

import mujoco
import numpy as np
from numpy import typing as npt

from .base_renderer import BaseRenderer


class OffscreenRenderer(BaseRenderer):
    def _initialize(self, width: Optional[int] = None, height: Optional[int] = None,
                    depth: bool = False, segmentation: bool = False) -> None:
        if depth and segmentation:
            raise ValueError('depth and segmentation options are mutually exclusive')

        width = width or self._model.vis.global_.offwidth
        height = height or self._model.vis.global_.offheight

        self.viewer = mujoco.Renderer(self._model, height, width, self._model.ngeom * 2)

        if depth:
            self.viewer.enable_depth_rendering()
        elif segmentation:
            self.viewer.enable_segmentation_rendering()

        self._camera = mujoco.MjvCamera()
        self._options = mujoco.MjvOption()

        # set camera
        self.camera.fixedcamid = self._camera_id
        if self._camera_id != -1:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

    @property
    def scene(self) -> mujoco.MjvScene:
        return self.viewer.scene

    @property
    def camera(self) -> mujoco.MjvCamera:
        return self._camera

    @property
    def options(self) -> mujoco.MjvOption:
        return self._options

    @property
    def perturbations(self) -> None:
        return None  # no mouse perturbations in offscreen rendering

    def render(self, *, out: Optional[np.ndarray] = None) -> npt.NDArray:
        self.viewer.update_scene(self._data, self.camera, self.options)
        return self.viewer.render(out=out)

    def close(self) -> None:
        del self.viewer
