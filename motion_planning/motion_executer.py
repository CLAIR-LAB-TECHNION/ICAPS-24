import numpy as np

from .motion_planner import NTableBlocksWorldMotionPlanner

FACING_DOWN_R = [[0, 0, -1],
                 [0, 1, 0],
                 [1, 0, 0]]


class NTableBlocksWorldMotionExecuter:
    def __init__(self, env):
        self.env = env
        self.motion_planner = NTableBlocksWorldMotionPlanner()

        state = self.env.get_state()

        # set current configuration
        self.motion_planner.set_config(state['robot_joint_pos'])

        # update world model with blocks
        for name, pos in state['object_positions'].items():
            self.motion_planner.add_block(name, pos)

    def move_to(self, target_config, tolerance=0.05, end_vel=0.1, max_steps=None,
                render_freq=8):
        '''
        move robot joints to target config, until it is close within tolerance,
        or max_steps exceeded.
        @param target_joint_pos: position to move to
        @param tolerance: distance withing configuration space to target to consider
         as reached
        @param max_steps: maximum steps to take before stopping, no limit if None
        @param render_freq: how often to render and append a frame
        @return: success, frames
        '''
        joint_positions = self.env.robot_joint_pos
        joint_velocities = self.env.robot_joint_velocities

        frames = []

        i = 0
        while np.linalg.norm(joint_positions - target_config) > tolerance \
                or np.linalg.norm(joint_velocities) > end_vel:
            if max_steps is not None and i > max_steps:
                return False, frames

            state = self.env.step(target_config)
            joint_positions = state['robot_joint_pos']
            joint_velocities = state['robot_joint_velocities']

            if i % render_freq == 0:
                frames.append(self.env.render())

            i += 1

        return True, frames

    def update_blocks_positions(self):
        blocks_positions_dict= self.env.get_state()['object_positions']
        for name, pos in blocks_positions_dict.items():
            self.motion_planner.move_block(name, pos)

    def execute_path(self, path, tolerance=0.05, end_vel=0.1,
                     max_steps_per_section=200, render_freq=8):
        """
        execute a path of joint positions
        @param path: list of joint positions to follow
        @param tolerance: distance withing configuration space to target to consider as reached to each point
        @param end_vel: maximum velocity to consider as reached to each point
        @param max_steps_per_section: maximum steps to take before stopping at each section
        @param render_freq: how often to render and append a frame
        @return: success, frames
        """
        frames = []
        for config in path:
            success, frames_curr = self.move_to(
                                           config,
                                           tolerance=tolerance,
                                           end_vel=end_vel,
                                           max_steps=max_steps_per_section,
                                           render_freq=render_freq)
            frames.extend(frames_curr)
            if not success:
                return False, frames

        return True, frames

    def move_to_pose(self, target_position,
                     target_orientation=FACING_DOWN_R, tolerance=0.05,
                     end_vel=0.1, max_steps_per_section=400, render_freq=8):
        """
        move robot to target position and orientation, and update the motion planner
        with the new state of the blocks
        @param target_position: position to move to
        @param target_orientation: orientation to move to
        @param tolerance: distance withing configuration space to target to consider as reached
        @param max_steps_per_section: maximum steps to take before stopping a section
        @param render_freq: how often to render and append a frame
        @return: success, frames
        """
        joint_state = self.env.robot_joint_pos
        path = self.motion_planner.plan_from_config_to_pose(joint_state, target_position,
                                                       target_orientation)
        success, frames = self.execute_path(
                                       path,
                                       tolerance=tolerance,
                                       end_vel=end_vel,
                                       max_steps_per_section=max_steps_per_section,
                                       render_freq=render_freq)

        # after executing a motion, blocks position can change, update the motion planner:
        self.update_blocks_positions()

        return success, frames

    def move_above_block(self, block_name, offset=0.1,
                         tolerance=0.05, end_vel=0.1, max_steps_per_section=400,
                         render_freq=8):
        """
        move robot above a block
        @param block_name: name of the block to move above
        @param offset: how much above the block to move
        @param tolerance: distance withing configuration space to target to consider as reached
        @param max_steps_per_section: maximum steps to take before stopping a section
        @param render_freq: how often to render and append a frame
        @return: success, frames
        """
        block_pos = self.env.get_object_pos(block_name)
        target_position = block_pos + np.array([0, 0, offset])
        return self.move_to_pose(target_position,
                            tolerance=tolerance, end_vel=end_vel,
                            max_steps_per_section=max_steps_per_section,
                            render_freq=render_freq)

    def activate_grasp(self, wait_steps=10, render_freq=8):
        self.env.set_gripper(True)
        self.motion_planner.attach_box_to_ee()
        return self.wait(wait_steps, render_freq=render_freq)

    def deactivate_grasp(self, wait_steps=10, render_freq=8):
        self.env.set_gripper(False)
        self.motion_planner.detach_box_from_ee()
        return self.wait(wait_steps, render_freq=render_freq)

    def wait(self, n_steps, render_freq=8):
        frames = []

        # move to current position, i.e., stay in place
        maintain_pos = self.env.robot_joint_pos
        for i in range(n_steps):
            self.env.step(maintain_pos)
            if i % render_freq == 0:
                frames.append(self.env.render())

        # account for falling objects
        self.update_blocks_positions()

        return True, frames
