import numpy as np


class ObjectManager:
    """convience class to manage graspable objects in the mujoco simulation"""

    def __init__(self, mj_model, mj_data):
        self._mj_model = mj_model
        self._mj_data = mj_data

        # manipulated objects have 6dof free joint that must be named in the mcjf.
        all_joint_names = [self._mj_model.joint(i).name for i in range(self._mj_model.njnt)]

        # all bodies that ends with "box"
        self.object_names = [name for name in all_joint_names if name.startswith("block")]
        self.objects_mjdata_dict = {name: self._mj_model.joint(name) for name in self.object_names}

        self.initial_positions_dict = self.get_all_object_positons_dict()

    def reset_object_positions(self):
        for name, pos in self.initial_positions_dict.items():
            self.set_object_pose(name, pos,[0, 1, 0, 0])

    def get_all_object_positons_dict(self):
        return {name: self.get_object_pos(name) for name in self.object_names}

    def get_object_pos(self, name: str):
        return self._mj_data.joint(name).qpos[:3]

    def get_object_quat(self, name: str):
        return self._mj_data.joint(name).qpos[3:7]

    def set_object_pose(self, name: str, pos, quat):
        joint_id = self.objects_mjdata_dict[name].id
        pos_adr = self._mj_model.jnt_qposadr[joint_id]
        self._mj_data.qpos[pos_adr:pos_adr + 7] = np.concatenate([pos, quat])

    def set_object_vel(self, name: str, cvel):
        joint_id = self.objects_mjdata_dict[name].id
        vel_adr = self._mj_model.jnt_dofadr[joint_id]
        self._mj_data.qvel[vel_adr:vel_adr + 6] = cvel
