import time
import numpy as np
from klampt import WorldModel, Geometry3D, RobotModel
from klampt.model.geometry import box
from klampt import vis
from klampt.plan.cspace import MotionPlan
from klampt.plan.robotcspace import RobotCSpace
from klampt.model.collide import WorldCollider
from klampt.plan import robotplanning
from klampt.model import ik
from n_table_blocks_world.configurations_and_constants import *
from motion_planning.configurations import limits_l, limits_h, default_config


class NTableBlocksWorldMotionPlanner():
    def __init__(self, eps=1e-2):
        """
        parameters:
        eps: epsilon gap for collision checking along the line in configuration space. Too high value may lead to
            collision, too low value may lead to slow planning. Default value is 1e-2.
        """
        self.eps = eps

        self.world = WorldModel()
        self.world.readFile("./motion_planning/klampt_world.xml")
        self._build_world()

        self.robot = self.world.robot(0)
        self.ee_link = self.robot.link("ee_link")

        # values are imported from configuration file
        self.robot.setJointLimits(limits_l, limits_h)
        self.planning_config = default_config

    def plan_from_config_to_pose(self, start_config, goal_pos, goal_R, max_time=30, max_length_to_distance_ratio=2):
        """
        plan from a start configuration to a goal pose that is given in 6d configuration space
        @param start_config: 6d configuration
        @param goal_pos: ee position
        @param goal_R: ee orientation as a rotation matrix
        @param max_time: maximum planning time
        @return: path in 6d configuration space
        """
        start_config_klampt = self.config6d_to_klampt(start_config)
        path = self._plan_from_config_to_pose_klampt(start_config_klampt, goal_pos, goal_R, max_time,
                                                     max_length_to_distance_ratio)

        return self.path_klampt_to_config6d(path)

    def plan_from_start_to_goal_config(self, start_config, goal_config, max_time=30, max_length_to_distance_ratio=2):
        """
        plan from a start and a goal that are given in 6d configuration space
        """
        start_config_klampt = self.config6d_to_klampt(start_config)
        goal_config_klampt = self.config6d_to_klampt(goal_config)
        path = self._plan_from_start_to_goal_config_klampt(start_config_klampt, goal_config_klampt, max_time,
                                                           max_length_to_distance_ratio)

        return self.path_klampt_to_config6d(path)

    def ik_solve(self, goal_pos, goal_R, start_config=None):
        """
        solve inverse kinematics to find a configuration that reaches the goal pose
        @param goal_pos: ee position
        @param goal_R: ee orientation as a rotation matrix
        @param start_config: initial guess for the IK solver
        @return: 6d configuration
        """
        if start_config is not None:
            start_config = self.config6d_to_klampt(start_config)
        return self.klampt_to_config6d(self._ik_solve_klampt(goal_pos, goal_R, start_config))

    def config6d_to_klampt(self, config):
        """
        There are 10 links in our URDF for klampt, some are stationary, actual joints are 2:8
        """
        config_klampt = [0] * 10
        config_klampt[2:8] = config
        return config_klampt

    def klampt_to_config6d(self, config_klampt):
        """
        There are 10 links in our URDF for klampt, some are stationary, actual joints are 2:8
        """
        return config_klampt[2:8]

    def path_klampt_to_config6d(self, path_klampt):
        """
        convert a path in klampt 10d configuration space to 6d configuration space
        """
        if path_klampt is None:
            return None
        path = []
        for q in path_klampt:
            path.append(self.klampt_to_config6d(q))
        return path

    def open_vis(self):
        """
        open visualization window
        """
        vis.add("world", self.world)
        # set camera position:
        viewport = vis.getViewport()
        viewport.camera.tgt = [0, 0, 0.7]
        viewport.camera.rot = [0, -0.7, 2]

        vis.show()
        pass

    def show_path_vis(self, path):
        """
        show the path in the visualization
        """
        if len(path[0]) == 6:
            path = [self.config6d_to_klampt(q) for q in path]
        self.robot.setConfig(path[0])
        vis.add("path", path)
        vis.setColor("path", 0, 1, 0, 1)

    def move_to_config_vis(self, q):
        """
        move to a config wwithin the internal world model of the motion planner. Use for visualization.
        """
        if len(q) == 6:
            q = self.config6d_to_klampt(q)  # convert to klampt 10d config
        self.robot.setConfig(q)

    def vis_spin(self, t):
        """
        spin the visualization for n steps
        """
        vis.spin(t)

    def _ik_solve_klampt(self, goal_pos, goal_R, start_config=None):
        """
        solve inverse kinematics to find a configuration that reaches the goal pose
        @param goal_pos: ee position
        @param goal_R: ee orientation as a rotation matrix
        @param start_config: initial guess for the IK solver
        @return: 6d configuration
        """
        if start_config is not None:
            self.robot.setConfig(start_config)
        goal_R = np.array(goal_R).flatten()  # flatten the 3x3 rotation matrix, as needed for klampt
        ik_objective = ik.objective(self.ee_link, R=goal_R, t=goal_pos)
        if not ik.solve_nearby(ik_objective, 1):
            print("no ik solution found")
        return self.robot.getConfig()

    def _plan_from_config_to_pose_klampt(self, start_config, goal_pos, goal_R, max_time=30,
                                         max_length_to_distance_ratio=2):
        """
        plan from a start configuration to a goal pose that is given in klampt 10d configuration space
        @param start_config: 10d configuration
        @param goal_pos: ee position
        @param goal_R: ee orientation as a rotation matrix
        @param max_time: maximum planning time
        @return: path in 10d configuration space
        """
        self.robot.setConfig(start_config)

        R = np.array(goal_R).flatten()  # flatten the 3x3 rotation matrix, as needed for klampt
        goal_objective = ik.objective(self.ee_link, R=R, t=goal_pos)

        planner = robotplanning.plan_to_cartesian_objective(self.world, self.robot, [goal_objective],
                                                            # extraConstraints=[space_reduction_constraint],
                                                            **self.planning_config)
        planner.space.eps = self.eps
        return self._plan(planner, max_time, max_length_to_distance_ratio)

    def _plan_from_start_to_goal_config_klampt(self, start_config, goal_config, max_time=30,
                                               max_length_to_distance_ratio=2):
        """
        plan from a start and a goal that are given in klampt 10d configuration space
        """
        self.robot.setConfig(start_config)

        planner = robotplanning.plan_to_config(self.world, self.robot, goal_config,
                                               # extraConstraints=[space_reduction_constraint],
                                               **self.planning_config)
        planner.space.eps = self.eps
        return self._plan(planner, max_time, max_length_to_distance_ratio)

    def _plan(self, planner: MotionPlan, max_time=30, steps_per_iter=100, max_length_to_distance_ratio=2):
        """
        find path given a prepared planner, with endpoints already set
        @param planner: MotionPlan object, endpoints already set
        @param max_time: maximum planning time
        @param steps_per_iter: steps per iteration
        @param max_length_to_distance_ratio: maximum length of the pass to distance between start and goal. If there is
            still time, the planner will continue to plan until this ratio is reached. This is to avoid long paths
            where the robot just moves around because non-optimal paths are still possible.
        """
        start_time = time.time()
        path = None
        while (path is None or self.compute_path_length_to_distance_ratio(path) > max_length_to_distance_ratio) \
                and time.time() - start_time < max_time:
            print("planning...")
            planner.planMore(steps_per_iter)
            path = planner.getPath()
        if path is None:
            print("no path found")
        return path

    def compute_path_length(self, path):
        """
        compute the length of the path
        """
        if path is None:
            return np.inf
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
        return length

    def compute_path_length_to_distance_ratio(self, path):
        """ compute the ratio of path length to the distance between start and goal """
        if path is None:
            return np.inf
        start = np.array(path[0])
        goal = np.array(path[-1])
        distance = np.linalg.norm(start - goal)
        length = self.compute_path_length(path)
        return length / distance

    def _build_world(self):
        """ build the obstacles in the world """
        # all sizes and positions are imported from configuration file
        self._add_box_geom("floor", (5, 5, 0.01), [0, 0, 0], [0.1, 0.1, 0.1, 1])
        self._add_box_geom("table_left", table_size, table_left_pos, [0.5, 0.5, 0.5, 0.8])
        self._add_box_geom("table_right", table_size, table_right_pos, [0.5, 0.5, 0.5, 0.8])
        self._add_box_geom("table_front", table_size, table_front_pos, [0.5, 0.5, 0.5, 0.8])
        # self._add_box_geom("purpule_box", (0.1, 0.1, 0.02), [-0.2, 0.5, 0.72], [0.5, 0.1, 0.5, 1])
        self._add_box_geom("robot_mount_approx", size=(0.45, 0.25, mount_top_base),
                           center=[-0.1, 0, mount_top_base / 2], color=[0.5, 0.5, 0.5, 1])

    def _add_box_geom(self, name, size, center, color):
        """
        add box geometry for collision in the world
        """
        width, depth, height = size
        box_obj = box(width=width, height=height, depth=depth, center=center)
        box_geom = Geometry3D()
        box_geom.set(box_obj)
        box_rigid_obj = self.world.makeRigidObject(name)
        box_rigid_obj.geometry().set(box_geom)
        box_rigid_obj.appearance().setColor(*color)
