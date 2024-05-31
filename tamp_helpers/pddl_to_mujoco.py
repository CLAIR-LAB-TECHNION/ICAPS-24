import numpy as np


# NOTE: PDDL is case-insensitive, and therefore most parsers default to lowercase object names.
OBJECT_PDDL_ID_TO_MUJOCO_NAME = {
    # tables
    't1': 'table_left_top',  # these are specifically the names for the table tops, not the entire table
    't2': 'table_front_top',
    't3': 'table_right_top',

    # blocks
    'b1': 'block 1',
    'b2': 'block 2',
    'b3': 'block 3',
    'b4': 'block 4',
    'b5': 'block 5',
    'b6': 'block 6',
    'b7': 'block 7',
    'b8': 'block 8',
    'b9': 'block 9',
    'b10': 'block 10'
}

TABLE_PDDL_IDS = [
    pddl_id
    for pddl_id in OBJECT_PDDL_ID_TO_MUJOCO_NAME.keys()
    if pddl_id.startswith('t')
]
BLOCK_PDDL_IDS = [
    pddl_id
    for pddl_id in OBJECT_PDDL_ID_TO_MUJOCO_NAME.keys()
    if pddl_id.startswith('b')
]

COLOR_PDDL_ID_TO_RGBA_VALUE_IN_MUJOCO = {
    'r': np.array([1, 0, 0, 1]),
    'y': np.array([1, 1, 0, 1]),
    'c': np.array([0, 1, 1, 1]),
    'p': np.array([1, 0, 1, 1])
}


def pddl_id_to_mujoco_name(pddl_object_id):
  # the PDDL object ID is converted to lower case to support PDDL's case-insensitivity
  return OBJECT_PDDL_ID_TO_MUJOCO_NAME[pddl_object_id.lower()]


def pddl_id_to_mujoco_entity(object_id, env):
  # get mujoco identifier from PDDL ID
  object_name = pddl_id_to_mujoco_name(object_id)

  # get the entity associated with this object name
  return env._env.sim.get_entity(object_name, 'geom')


