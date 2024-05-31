import numpy as np
from pddlgymnasium.structs import Literal, Predicate, TypedEntity, Type

from n_table_blocks_world.physics_utils import get_normal_force


def entity_is_on_entity(entity1, entity2, state):
  """entity1 is considered "on" entity2 if the normal force from entity2 to entity1 is directly up"""
  # get normal force being applied by entity2 on entity1
  normal = get_normal_force(entity2, entity1, state)

  # calculate the distance between the normal force and the upward normal vector
  normal_dist_from_up_dir = np.linalg.norm(normal - np.array([0, 0, 1]))

  # determine whether the calculated distance is "close enough" to 0 using the `np.isclose` function
  # this function returns True if the distance we calculated is close to 0.
  # the default tolerance is 1e-08. this is too strict for us so we will change this to 0.1
  return np.isclose(normal_dist_from_up_dir, 0, atol=0.1)


def get_predicate_object(predicate_name, **vars):
  vars = [TypedEntity(name, Type(type_)) for name, type_ in vars.items()]
  pred = Predicate(predicate_name, len(vars))
  return Literal(pred, vars)

