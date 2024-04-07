import numpy as np


class ObjectManager:
    """convience class to manage graspable objects in the mujoco simulation"""

    def __init__(self, mj_model, mj_data):
        self._mj_model = mj_model
        self._mj_data = mj_data

        all_bodies_names = [self._mj_model.body(i).name for i in range(self._mj_model.nbody)]

        # all bodies that ends with "box"
        self.object_names = [name for name in all_bodies_names if name.endswith("box")]
        self.objects_jntadrs_dict = {name: self._mj_model.body(name).jntadr[0] for name in self.object_names}

    def get_object_pos(self, name: str):
        return self._mj_data.body(name).xpos

    def get_object_quat(self, name: str):
        return self._mj_data.body(name).xquat

    def set_object_pose(self, name: str, pos, quat):
        adr = self.objects_jntadrs_dict[name]
        self._mj_data.qpos[adr:adr + 7] = np.concatenate([pos, quat])

