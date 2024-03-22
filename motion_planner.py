import time

import numpy as np
from klampt import WorldModel, Geometry3D, RobotModel
from klampt.model.geometry import box
from klampt import vis
from klampt.plan.cspace import MotionPlan
from klampt.plan.robotcspace import RobotCSpace
from klampt.model.collide import WorldCollider
from klampt.sim import Simulator
from klampt.model import ik


class TableWorldMotionPlanner():
    robot_height = 0.903 + 0.163 - 0.089159
    # 0.903 is the height of the robot mount, 0.163 is the height of the shift of shoulder link in mujoco,
    # 0.089159 is the height of shoulder link in urdf for klampt
    mount_top_base = robot_height - 0.01  # avoid collision between robot base and mount

    def __init__(self, eps=1e-2):
        """
        parameters:
        eps: epsilon gap for collision checking along the line in configuration space. Too high value may lead to
            collision, too low value may lead to slow planning. Default value is 1e-2.
        """
        self.world = WorldModel()
        self.world.readFile("klampt_world.xml")

        self.robot = self.world.robot(0)
        self.ee_link = self.robot.link("ee_link")
        self.collider = WorldCollider(self.world)
        self.cspace = RobotCSpace(self.robot, self.collider)
        self.cspace.eps = eps

        MotionPlan.setOptions(type="rrt", bidirectional=True, shortcut=True)
        self.planner = MotionPlan(self.cspace)

        self._build_world()

    def plan_from_start_to_goal_config(self, start_config, goal_config, max_time=30):
        """
        plan from a start and a goal that are given in 6d configuration space
        """
        start_config_klampt = self.config6d_to_klampt(start_config)
        goal_config_klampt = self.config6d_to_klampt(goal_config)
        path = self._plan_from_start_to_goal_config_klampt(start_config_klampt, goal_config_klampt, max_time)

        for i in range(len(path)):
            path[i] = self.klampt_to_config6d(path[i])
        return path


    def config6d_to_klampt(self, config):
        """
        There are 10 links in our URDF for klampt, some are stationary, actual joints are 2:8
        """
        config_klampt = [0]*10
        config_klampt[2:8] = config
        return config_klampt

    def klampt_to_config6d(self, config_klampt):
        """
        There are 10 links in our URDF for klampt, some are stationary, actual joints are 2:8
        """
        return config_klampt[2:8]

    def ik_solve_6d(self, goal_pos, goal_R):
        """
        find inverse kinematic solution for a given goal position and orientation. solution is a 6d vector,
        as used in our 6d configuration space
        """
        sol = self._ik_solve(goal_pos, goal_R)
        if sol is None:
            return None
        return self.klampt_to_config6d(sol)

    def open_vis(self):
        """
        open visualization window
        """
        vis.add("world", self.world)
        vis.show()

    def move_to_config_vis(self, q):
        """
        move to a config wwithin the internal world model of the motion planner. Use for visualization.
        """
        if len(q) == 6:
            q = self.config6d_to_klampt(q)  # convert to klampt 10d config
        self.robot.setConfig(q)

    def vis_spin(self, n):
        """
        spin the visualization for n steps
        """
        vis.spin(n)

    def _plan_from_start_to_goal_config_klampt(self, start_config, goal_config, max_time=30):
        """
        plan from a start and a goal that are given in klampt 10d configuration space
        """

        self.planner.setEndpoints(start_config, goal_config)

        start_time = time.time()
        path = None
        while path is None and time.time() - start_time < max_time:
            self.planner.planMore(50)
            path = self.planner.getPath()
        if path is None:
            print("no path found")
        return path

    def _ik_solve(self, goal_pos, goal_R):
        """
        find inverse kinematic solution for a given goal position and orientation. solution is a 10d vector,
        as usedfor klampt
        """
        goal_R_flat = np.array(goal_R).flatten()
        goal_objective = ik.objective(self.ee_link, R=goal_R_flat, t=goal_pos)
        if not ik.solve_global(goal_objective):
            print("no ik solution for goal: ", goal_pos, goal_R)
            return None
        sol = self.robot.getConfig()
        # check for collision in world and robot:
        if self.cspace.selfCollision() or any(self.collider.robotObjectCollisions(self.robot)):
            print("ik solution is in collision")
            return None

        return sol

    def _build_world(self):
        self._add_box_geom("floor", (5, 5, 0.01), [0, 0, 0], [0.1, 0.1, 0.1, 1])
        self._add_box_geom("table_left", (0.74, 0.74, 0.01), [0, -0.6, 0.7], [0.5, 0.5, 0.5, 0.8])
        self._add_box_geom("table_right", (0.74, 0.74, 0.01), [0, 0.6, 0.7], [0.5, 0.5, 0.5, 0.8])
        self._add_box_geom("purpule_box", (0.1, 0.1, 0.02), [-0.2, 0.5, 0.72], [0.5, 0.1, 0.5, 1])
        self._add_box_geom("robot_mount_approx", size=(0.45, 0.3, self.mount_top_base),
                           center=[-0.1, 0, self.mount_top_base / 2], color=[0.5, 0.5, 0.5, 1])

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
