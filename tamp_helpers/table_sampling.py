import numpy as np

from .pddl_to_mujoco import pddl_id_to_mujoco_entity, BLOCK_PDDL_IDS

def sample_on_table(table_id, env, pddl_to_mj=None, z_offset=0.2):
  """
  sample (x, y, z) coordinates that are located on the given table.
  """
  pddl_to_mj = pddl_to_mj or pddl_id_to_mujoco_entity

  # get mujoco actionable entity for the table
  table_entity = pddl_to_mj(table_id, env)

  # retrieve table info
  x_size, y_size, z_size = table_entity.size
  x, y, z = table_entity.center_of_mass

  # determine the range of allowed values in the x and y axes
  x_range = [x - x_size / 2, x + x_size / 2]
  y_range = [y - y_size / 2, y + y_size / 2]

  # z does not need to be sampled since the height of the table is consistent for all x,y values
  sample_z = z + z_size + z_offset
  sample_x, sample_y = np.random.uniform(*np.array([x_range, y_range]).T)

  # return coordinates as numpy array
  return np.array([sample_x, sample_y, sample_z])


def is_pos_colliding_with_block(block_id, sampled_pos, env, pddl_to_mj=None, padding=0.07):
  """
  determines whether placing a block in a sampled position will cause a collision with a specific block.
  """
  pddl_to_mj = pddl_to_mj or pddl_id_to_mujoco_entity

  # get mujoco actionable entity for the block with which to check collision
  block_entity = pddl_to_mj(block_id, env)

  # get block info
  block_size = block_entity.size + padding
  block_pos = block_entity.center_of_mass

  # low and high shift based on full block size to account for the block being checked AND the block
  # being placed in the sampled position (assuming the blocks have the same shape and size).
  block_low = block_pos - block_size
  block_high = block_pos + block_size

  # check only x-y axis
  return np.all(block_low[:2] < sampled_pos[:2]) and np.all(sampled_pos[:2] < block_high[:2])


def is_pos_colliding(block_id, sampled_pos, env, pddl_to_mj, ids, skip_self=True):
  """
  determines whether placing a block in a sampled position will cause a collision with
  another block.
  """
  ids = ids or BLOCK_PDDL_IDS

  # iterate all blocks
  for other_block_id in ids:

    # skip the block being placed
    if other_block_id == block_id and skip_self:
      continue

    # check for collision with the current block in the iteration
    if is_pos_colliding_with_block(other_block_id, sampled_pos, env, pddl_to_mj):
      return True

  # no collisions found
  return False


def sample_free_spot_on_table_for_block(table_id, block_id, env, pddl_to_mj=None, z_offset=0.2, ids=None, max_attempts=1_000, skip_self=True):
  """
  samples a spot on a given table to place a given block such that there is no collision
  with another block. Will sample `max_attempts` times before raising an error.
  """
  # sample first attempt
  attempt = 0
  sampled_pos = sample_on_table(table_id, env, pddl_to_mj, z_offset)

  # continue sampling until a good position is found or until the maximum number of attempts
  # is reached.
  while is_pos_colliding(block_id, sampled_pos, env, pddl_to_mj, ids, skip_self) and attempt < max_attempts:
    sampled_pos = sample_on_table(table_id, env, pddl_to_mj, z_offset)
    attempt += 1

  # check for failure based on number of attempts.
  if attempt >= max_attempts:
    raise TimeoutError(f'Could not find a collision-free space on table {table_id} after {attempt} attempts')

  return sampled_pos
