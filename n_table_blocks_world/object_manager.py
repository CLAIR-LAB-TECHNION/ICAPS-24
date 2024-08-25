import numpy as np


class ObjectManager:
    """convience class to manage graspable objects in the mujoco simulation"""

    def __init__(self, sim):
        self._mj_model = sim.model
        self._mj_data = sim.data

        self.object_names = []
        self.object_mjcfs = {}
        self.objects_mjdata_dict = {}
        self.immovable_objects = {}
        for spec in sim.scene.objects:
            obj = sim.composer.get_object(spec, as_attachment_element=True)
            obj_name = obj.full_identifier[:-1] if obj.full_identifier.endswith('/') else obj.full_identifier
            self.object_names.append(obj_name)
            self.object_mjcfs[obj_name] = obj

            joints = obj.find_all('joint')
            if len(joints) == 0:
                self.immovable_objects[obj_name] = obj
                continue
            base_joint = joints[0]
            self.objects_mjdata_dict[obj_name] = self._mj_data.joint(base_joint.full_identifier)

        self.initial_positions_dict = self.get_all_object_positons_dict()

    def reset_object_positions(self):
        for name, (pos, quat) in self.initial_positions_dict.items():
            if name in self.immovable_objects:
                continue
            self.set_object_pose(name, pos, quat)

    def get_all_object_positons_dict(self):
        object_poses = {name: (self.get_object_pos(name), self.get_object_quat(name)) for name in self.object_names
                        if name not in self.immovable_objects}
        object_poses.update({
            name: (obj.pos, [1, 0, 0, 0]) for name, obj in self.immovable_objects.items()
        })

        return object_poses

    def get_object_pos(self, name: str):
        return self.objects_mjdata_dict[name].qpos[:3]

    def get_object_quat(self, name: str):
        return self.objects_mjdata_dict[name].qpos[3:7]

    def set_object_pose(self, name: str, pos, quat):
        joint_id = self.objects_mjdata_dict[name].id
        pos_adr = self._mj_model.jnt_qposadr[joint_id]
        self._mj_data.qpos[pos_adr:pos_adr + 7] = np.concatenate([pos, quat])

    def set_object_vel(self, name: str, cvel):
        joint_id = self.objects_mjdata_dict[name].id
        vel_adr = self._mj_model.jnt_dofadr[joint_id]
        self._mj_data.qvel[vel_adr:vel_adr + 6] = cvel
