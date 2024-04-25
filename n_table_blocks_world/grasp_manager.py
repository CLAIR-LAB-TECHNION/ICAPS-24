import numpy as np
from n_table_blocks_world.object_manager import ObjectManager
from n_table_blocks_world.configurations_and_constants import *


class GraspManager():
    def __init__(self, mj_model, mj_data, object_manager: ObjectManager, min_grasp_distance=0.15):
        self._mj_model = mj_model
        self._mj_data = mj_data
        self.object_manager = object_manager
        self.min_grasp_distance = min_grasp_distance

        self.graspable_objects_names = object_manager.object_names

        self._ee_mj_data = self._mj_data.body('rethink_mount_stationary/ur5e/adhesive gripper/')

        self.attatched_object_name = None

    def grasp_nearest_object_if_close_enough(self) -> bool:
        """
        find the nearest object and grasp it if it is close enough
        """
        object_positions = [self.object_manager.get_object_pos(name) for name in self.graspable_objects_names]
        gripper_position = self._ee_mj_data.xpos

        # compute distances vectorized:
        distances = np.linalg.norm(np.array(object_positions) - gripper_position, axis=1)

        closest_object_idx = np.argmin(distances)
        distance = distances[closest_object_idx]

        if distance < self.min_grasp_distance:
            self.grasp_object(self.graspable_objects_names[closest_object_idx])
            return True
        else:
            return False

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

        target_position = self._ee_mj_data.xpos
        target_orientation = self._ee_mj_data.xquat
        target_velocities = self._ee_mj_data.cvel
        target_velocities = np.zeros(6)

        # add shift to target position to make sure object is a bit below end effector, but in ee frame
        target_position_in_ee = np.array([0, 0, grasp_offset])
        target_position = target_position + self._ee_mj_data.xmat.reshape(3, 3).T @ target_position_in_ee

        self.object_manager.set_object_pose(self.attatched_object_name, target_position, target_orientation)
        self.object_manager.set_object_vel(self.attatched_object_name, target_velocities)
