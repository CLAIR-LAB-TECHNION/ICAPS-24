import numpy as np


# NOTE: PDDL is case-insensitive, and therefore most parsers default to lowercase object names.
OBJECT_PDDL_ID_TO_MUJOCO_NAME = {
    # tables
    'wood_table': 'table_left_top',  # these are specifically the names for the table tops, not the entire table
    'white_table': 'table_front_top',
    'black_table': 'table_right_top',

    # blocks
    'red_block': 'cubeA',
    'green_block': 'cubeB',
    'blue_block': 'cubeC',
    'yellow_block': 'cubeD',
    'cyan_block': 'cubeE',
    'purple_block': 'cubeF',
}

TABLE_PDDL_IDS = [
    pddl_id
    for pddl_id in OBJECT_PDDL_ID_TO_MUJOCO_NAME.keys()
    if pddl_id.endswith('table')
]
BLOCK_PDDL_IDS = [
    pddl_id
    for pddl_id in OBJECT_PDDL_ID_TO_MUJOCO_NAME.keys()
    if pddl_id.endswith('block')
]


def pddl_id_to_mujoco_name(pddl_object_id):
  # the PDDL object ID is converted to lower case to support PDDL's case-insensitivity
  return OBJECT_PDDL_ID_TO_MUJOCO_NAME[pddl_object_id.lower()]


def pddl_id_to_mujoco_entity(object_id, env):
  # get mujoco identifier from PDDL ID
  object_name = pddl_id_to_mujoco_name(object_id)

  # get the entity associated with this object name
  return env._env.sim.get_entity(object_name, 'geom')


