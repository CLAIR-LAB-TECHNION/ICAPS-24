import numpy as np


def get_contacts(geom1, geom2, state):
    """
    get the indices in the MuJoCo contacts database that match those of the given geoms, and the direction
    of contact (1 - geom1 to geom2; 2- geom2 to geom1)
    """
    contact_idx = []
    contact_dir = []
    for i, (contact_id1, contact_id2) in enumerate(state['geom_contact'].geom):
        if contact_id1 == geom1.element_id and contact_id2 == geom2.element_id:
            contact_idx.append(i)
            contact_dir.append(1)
        elif contact_id2 == geom1.element_id and contact_id1 == geom2.element_id:
            contact_idx.append(i)
            contact_dir.append(-1)

    return contact_idx, contact_dir


def get_normal_force(geom1, geom2, state):
    """
    get the normal force applied by geom1 on geom2
    """
    geom_contacts, geom_contact_dirs = get_contacts(geom1, geom2, state)

    if not geom_contacts:
        return np.array([0, 0, 0])

    frame = state['geom_contact'].frame[geom_contacts]

    # normal force is index 0-2 of the force frame (Z-axis). direction is always geom1 to geom2
    # see https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h
    all_contact_normals = frame.T[0:3]  # transpose to enable referencing (x,y,z) at the top level
    all_contact_normals = all_contact_normals * geom_contact_dirs  # set direction according to args order

    # average all contact normals for one definite normal force
    return all_contact_normals.mean(axis=1)
