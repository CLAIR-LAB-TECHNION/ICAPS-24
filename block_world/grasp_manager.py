import numpy as np
from block_world.object_manager import ObjectManager


class GraspManager():
    def __init__(self, mj_model, mj_data, object_manager: ObjectManager):
        self._mj_model = mj_model
        self._mj_data = mj_data
        self.object_manager = object_manager

        self.graspable_objects_names = object_manager.object_names

        self._ee_mj_data = self._mj_data.body('rethink_mount_stationary/ur5e/adhesive gripper/')

        self.attatched_object_name = None

    def grasp_nearest_object_if_close_enough(self):
        """
        find the nearest object and grasp it if it is close enough
        """
        pass

    def grasp_object(self, object_name):
        """
        attatch this object to the gripper position
        """
        self.attatched_object_name = object_name
        self.update_grasped_object_pose()

    def release_object(self):
        """
        release the object from the gripper
        """
        self.attatched_object_name = None

    def update_grasped_object_pose(self):
        """
        update the pose of the object that is currently grasped to be on the gripper
        """
        if self.attatched_object_name is None:
            return

        object_jntadr = self._mj_model.body(self.attatched_object_name).jntadr[0]

        target_position = self._ee_mj_data.xpos
        target_orientation = self._ee_mj_data.xquat

        self.object_manager.set_object_pose(self.attatched_object_name, target_position, target_orientation)
