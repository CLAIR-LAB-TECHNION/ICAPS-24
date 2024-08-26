import numpy as np

from motion_planning.motion_executer import NTableBlocksWorldMotionExecuter
from .pddl_to_mujoco import pddl_id_to_mujoco_name
from .table_sampling import sample_free_spot_on_table_for_block



home_config = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

class SkillExecuter:
  def __init__(self, env, render_freq=0):
      self.exec = NTableBlocksWorldMotionExecuter(env)
      self.render_freq = render_freq

  def pick_up(self, block_id, table_id):
    # NOTE: we accept the table_id parameter but don't use it.
    # we do not need this parameter because we already have the position of each block from the simulator.
    # we keep the argument for compatibility with the PDDL action.

    # get block identifier in MuJoCo
    block_name = pddl_id_to_mujoco_name(block_id)

    # move end-effector above the block
    move_suc, move_frames = self.exec.move_above_block(block_name, render_freq=self.render_freq)

    # activate the gripper to grasp the object
    grasp_suc, grasp_frames = self.exec.activate_grasp(render_freq=self.render_freq)

    return move_suc and grasp_suc, np.concatenate([move_frames, grasp_frames])

  def put_down(self, block_id, table_id):
    # sample a collision free spot on the table
    table_pos = sample_free_spot_on_table_for_block(table_id, block_id, self.exec.env)

    # move end-effector with the block above the sampled spot on the table.
    move_suc, move_frames = self.exec.move_to_pose(table_pos, render_freq=self.render_freq)

    # deactivate the girpper to release the object
    grasp_suc, grasp_frames = self.exec.deactivate_grasp(render_freq=self.render_freq)

    return move_suc and grasp_suc, np.concatenate([move_frames, grasp_frames])

  def stack(self, block1_id, block2_id):
    # get identifier in MuJoCo for the block on which to stack
    block2_name = pddl_id_to_mujoco_name(block2_id)

    # move end-effector with the destination block
    move_suc, move_frames = self.exec.move_above_block(block2_name, render_freq=self.render_freq)

    # deactivate the girpper to release the object
    grasp_suc, grasp_frames = self.exec.deactivate_grasp(render_freq=self.render_freq)

    return move_suc and grasp_suc, np.concatenate([move_frames, grasp_frames])

  def unstack(self, block1_id, block2_id):
    # we can just pick up the block from its location without considering the other block or the table it is on.
    return self.pick_up(block1_id, None)  # remember, we are ignoring the table ID in the pick_up method.

  def go_home(self):
    # a simple method to move to the robot arm's home position
    return self.exec.move_to(home_config)


def execute_plan(env, best_node, estimator, executer):
  suc, frames = True, []

  for state, action, next_state in best_node.get_transition_path():
    # no action for initial state transition
    if action is not None:
      # execute action
      action = action.content

      # =======================
      # STATE SAMPLER USED HERE
      # =======================
      skill = getattr(executer, action.predicate.name.replace('-', '_'))
      args = vars = list(map(lambda v: v.name, action.variables))
      skill_suc, skill_frames = skill(*args)

      suc &= skill_suc
      frames.append(skill_frames)

    # check that we have reached the desired state after the transition
    # the first time this will be the initial state
    desired_cur_state = next_state.content
    world_state = env.get_state()

    # =========================
    # STATE ESTIMATOR USED HERE
    # =========================
    cur_state_est = estimator(world_state)
    if not cur_state_est == desired_cur_state:
      # get literals in expected state that don't appear in the current state estimation
      expected_lits = desired_cur_state.literals.difference(cur_state_est.literals)
      expected_lits_msg = f'expected literals not in actual state:\n{expected_lits}'

      # get literals in the current state estimation that don't appear in the expected state
      actual_lits = cur_state_est.literals.difference(desired_cur_state.literals)
      actual_lits_msg = f'actual literals not in expected state:\n{actual_lits}'

      # raise failure error
      error = ValueError(f'Acheived wrong state.\n\n{expected_lits_msg}\n\n{actual_lits_msg}')
      if frames:  # append frames to error to allow rendering if exception is caught
        error.frames = np.concatenate(frames)
      else:
        error.frames = None
      raise error

  home_suc, home_frames = executer.go_home()
  suc &= home_suc
  frames.append(home_frames)

  return suc, np.concatenate(frames)


def execute_tamp_with_replanning(env, problem, solver, estimator, executer, max_retries=10):
  env.reset()

  trials = 0
  best_node = None
  frames_agg = []

  # run while:
  # this is the first trial OR
  # a high-level task plan is found AND trial limit is not reached
  while (best_node is not None or trials == 0) and trials < max_retries:
    print(f'planning trial {trials + 1}/{max_retries}')

    # get high-level state estimation from current world state
    env_state = env.get_state()
    estimated_pddl_state = estimator(env_state)

    # set the estimated high-level state as the current state of the PDDL problem
    problem.env.set_state(estimated_pddl_state)
    problem.initial_state = estimated_pddl_state

    # solve the problem using the given solver
    best_node = solver(problem)

    # stop execution if no plan exists
    if best_node is None:
        break

    try:
      # execute plan
      suc, frames = execute_plan(env, best_node, estimator, executer)
      frames_agg.append(frames)
      break
    except ValueError as e:
      # plan execution failed. go to next trial
      print(f'\ntask plan execution failed with error:\n{e.args[0]}\n')
      if e.frames is not None:
        frames_agg.append(e.frames)

      if trials < max_retries:
        print('attempting to replan')

    trials += 1

  # determine run success
  success = best_node is not None and trials < max_retries

  # print success message
  suc_msg = f'TAMP execution ended with {f"success on trial {trials + 1}/{max_retries}" if success else "failure"}'
  print()
  print('=' * len(suc_msg))
  print(suc_msg)
  print('=' * len(suc_msg))

  # return success flag and render frames
  return success, np.concatenate(frames_agg)
