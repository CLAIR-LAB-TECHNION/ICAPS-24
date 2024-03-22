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


ROBOT_HEIGHT = 0.903 + 0.163 - 0.089159
# 0.903 is the height of the robot mount, 0.163 is the height of the shift of shoulder link in mujoco,
# 0.089159 is the height of shoulder link in urdf for klampt
MOUNT_TOP_BASE = ROBOT_HEIGHT - 0.01  # avoid collision between robot base and mount


def add_box_geom(world, name, size, center, color):
    width, depth, height = size
    box_obj = box(width=width, height=height, depth=depth, center=center)
    box_geom = Geometry3D()
    box_geom.set(box_obj)
    box_rigid_obj = world.makeRigidObject(name)
    box_rigid_obj.geometry().set(box_geom)
    box_rigid_obj.appearance().setColor(*color)

def joint_state_for_klampt(robot, q):
    q_klampt = np.zeros(robot.numLinks())
    for i, j in enumerate(robot.links):
        q_klampt[i] = q[j.index]
    return q_klampt


def set_joint_state(robot, angles):
    joints = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
    joint_ids = [robot.link(j).index for j in joints]

    q_klampt = joint_state_for_klampt(robot, angles)
    robot.setConfig(q_klampt)


if __name__ == '__main__':


    world = WorldModel()
    world.readFile("klampt_world.xml")

    add_box_geom(world, "floor", size=(5, 5, 0.01), center=[0, 0, 0], color=[0.1, 0.1, 0.1, 1])
    add_box_geom(world, "table_left", size=(0.74, 0.74, 0.01), center=[0, -0.6, 0.7], color=[0.5, 0.5, 0.5, 0.8])
    add_box_geom(world, "table_right", size=(0.74, 0.74, 0.01), center=[0, 0.6, 0.7], color=[0.5, 0.5, 0.5, 0.8])
    add_box_geom(world, "purpule_box", size=(0.1, 0.1, 0.02), center=[-0.2, 0.5, 0.72], color=[0.5, 0.1, 0.5, 1])
    add_box_geom(world, "robot_mount_approx", size=(0.45, 0.3, MOUNT_TOP_BASE), center=[-0.1, 0, MOUNT_TOP_BASE/2],
                 color=[0.5, 0.5, 0.5, 1])

    # set robot position in the world:
    robot = world.robot(0)

    collider = WorldCollider(world)
    # sim = Simulator(world)
    cspace = RobotCSpace(robot, collider)
    cspace.eps = 1e-2
    # cspace.setup()
    MotionPlan.setOptions(type="rrt", bidirectional=True, shortcut=True,)
    planner = MotionPlan(cspace)

    vis.add("world", world)
    vis.show()

    start_c = robot.getConfig()
    # can set config then get ee
    ee_link = robot.link("ee_link")
    start_T = robot.link("ee_link").getTransform()
    # T is tuple of (R, t) where R is flattened
    # move in y towards the base:
    goal_T = list(start_T)
    goal_T[1][1] -= 0.4
    goal_objective = ik.objective(body=ee_link, R=goal_T[0], t=goal_T[1])
    if not ik.solve(goal_objective):
        print("no ik solution for goal")
    goal_c = robot.getConfig()

    robot.setConfig(start_c)
    # look at noself collision in URDF. robot is coliding with itself at zero config

    planner.setEndpoints(start_c, goal_c)

    t = time.time()
    path = None
    while path is None:
        print("planning...")
        planner.planMore(50)
        path = planner.getPath()

    print("planning took: ", time.time() - t)

    planner.space.close()
    planner.close()

    vis.spin(int(1e9))

    # TODO next: motion plan, then check alignment with mujoco

    pass

