from abc import ABC, abstractmethod
from typing import Optional

import mujoco
import numpy.typing as npt
import numpy as np

from ..common import Identifier


class BaseRenderer(ABC):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, camera: Identifier, *args, **kwargs):
        self._model = model
        self._data = data

        # convert camera name to id if necessary
        if isinstance(camera, str):
            try:
                model.camera(camera)
            except KeyError as e:
                valid_cameras = str(e).split('.')[1].strip()
                raise ValueError(f'camera "{camera}" not found. {valid_cameras}')
            self._camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        else:
            if camera < -1:
                raise ValueError('camera_id cannot be smaller than -1.')
            if camera >= model.ncam:
                raise ValueError(f'model has {model.ncam} fixed cameras. camera_id={camera} is invalid.')
            self._camera_id = camera


        self._markers = []

        self._initialize(*args, **kwargs)

    @abstractmethod
    def _initialize(self, *args, **kwargs) -> None:
        pass

    @property
    @abstractmethod
    def scene(self) -> mujoco.MjvScene:
        pass

    @property
    @abstractmethod
    def camera(self) -> mujoco.MjvCamera:
        pass

    @property
    @abstractmethod
    def options(self) -> mujoco.MjvOption:
        pass

    @property
    @abstractmethod
    def perturbations(self) -> Optional[mujoco.MjvPerturb]:
        pass

    @abstractmethod
    def render(self) -> Optional[npt.NDArray]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
