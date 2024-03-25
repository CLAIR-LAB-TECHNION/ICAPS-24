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

from klampt.plan import robotplanning


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

    # collider = WorldCollider(world)
    # sim = Simulator(world)
    # cspace = RobotCSpace(robot, collider)
    # cspace.eps = 1e-3
    # cspace.setup()
    # MotionPlan.setOptions(type="rrt", bidirectional=True, shortcut=True,)

    vis.add("world", world)
    vis.show()
    time.sleep(1)

    start_c = robot.getConfig()
    # can set config then get ee
    ee_link = robot.link("ee_link")
    start_T = robot.link("ee_link").getTransform()
    # T is tuple of (R, t) where R is flattened
    # move in y towards the base:
    # goal_T = list(start_T)
    # goal_T[1][1] -= 0.4
    goal_T_1 = [[0, 0, -1, 0, 1, 0, 1, 0, 0], [-0.19144999999599827, 0.4172500000009377, 0.97135]]
    # same as 1 but rotate around z by 90 degrees:
    goal_T_2 = [[0, -1, 0, 0, 0, 1, -1, 0, 0], [-0.19144999999599827, 0.4172500000009377, 0.97135]]


    goal_objective_1 = ik.objective(body=ee_link, R=goal_T_1[0], t=goal_T_1[1])
    goal_objective_2 = ik.objective(body=ee_link, R=goal_T_2[0], t=goal_T_2[1])

    goal_objective = [goal_objective_1, ]

    # TODO: move to this : plan options dict is argument, add also cspace planning option
    # TODO: plan to multiple objectives (90 degrees rotation) and stop when one is reached, need to add constraints
    # TODO:     Use OR constraints example from chatgpt below commented
    t = time.time()
    mp = robotplanning.plan_to_cartesian_objective(world, robot, goal_objective,
                                                   type="rrt*", bidirectional=True,)
    mp.space.eps = 1e-2
    path = None
    while path is None:
        print("planning...")
        mp.planMore(100)
        # check if one of the goals is reached
        path = mp.getPath()
    print("path:")
    for p in path:
        print(p)
    print("planning took: ", time.time() - t)
    # vis.add("path", path)
    # vis.setColor("path", 0, 1, 0, 1)

    vis.spin(2)
    for p in path:
        robot.setConfig(p)
        vis.spin(1)

    vis.spin(int(1e9))

    # TODO next: motion plan, then check alignment with mujoco

    pass

"""
PLANNING with constraints example from chatgpt:


from klampt import robotplanning
from klampt import ik

# Define your target Cartesian position
target_position = [x, y, z]  # Replace x, y, z with your desired position coordinates

# Define orientation constraints for four rotation options around Z-axis (0, 90, 180, 270 degrees)
orientation_constraints = [
    ik.constraints.RotationConeConstraint(orientation=[0, 0, 1], angle=5),  # Option 1: within 5 degrees of Z-axis
    ik.constraints.RotationConeConstraint(orientation=[0, 0, 1], angle=5, axis=[0, 0, 1]),  # Option 2: within 5 degrees around Z-axis
    ik.constraints.RotationConeConstraint(orientation=[0, 0, 1], angle=5, axis=[0, 0, -1]),  # Option 3: within 5 degrees around -Z-axis
    ik.constraints.RotationConeConstraint(orientation=[0, 0, 1], angle=5, axis=[0, 0, 1]),  # Option 4: within 5 degrees around Z-axis
]

# Combine orientation constraints using logical "OR" operator
composite_orientation_constraint = ik.constraints.CustomConstraint()
composite_orientation_constraint.addTerms(orientation_constraints, operator='OR')

# Create the IK objective with position and composite orientation constraint
ik_objective = ik.objective.CustomIKObjective()
ik_objective.setFixedPoint(link, localpos=target_position)
ik_objective.addConstraint(composite_orientation_constraint)

# Define the list of IK objectives for the motion planner
ik_objectives = [ik_objective]

try:
    # Plan motion with IK constraints
    result = robotplanning.plan_to_cartesian_objective(world, robot, ik_objectives)
    
    # Check if planning succeeded
    if result is not None:
        print("Motion planning successful.")
    else:
        print("Failed to plan motion.")
except Exception as e:
    print(f"Motion planning error: {e}")
"""
