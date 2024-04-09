import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

from . import Entity
from ..episode.specs.camera_spec import CameraSpec
from .simulator import Simulator
from ..rendering.offscreen_renderer import OffscreenRenderer


class Camera:
    def __init__(self, camera_spec: CameraSpec, sim: Simulator):
        self.spec = camera_spec
        self.entity = Entity.from_name_and_tag(camera_spec.identifier, 'camera', sim.mjcf_model, sim.physics)
        self.viewer = OffscreenRenderer(sim.model, sim.data, self.entity.identifier,
                                        width=camera_spec.width, height=camera_spec.height,
                                        depth=camera_spec.depth, segmentation=camera_spec.segmentation)

    @property
    def name(self):
        identifier = self.entity.identifier
        dims = f'{self.spec.width}X{self.spec.height}'
        img_type = 'segmentation' if self.segmentation else 'depth' if self.depth else 'rgb'

        return f'camera_{identifier}_{dims}_{img_type}'

    def render(self):
        return self.viewer.render()

    def close(self):
        self.viewer.close()
