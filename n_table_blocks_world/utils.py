from collections import namedtuple


def convert_mj_struct_to_namedtuple(mj_struct):
    """
    convert a mujoco struct to a dictionary
    """
    attrs = [attr for attr in dir(mj_struct) if not attr.startswith('__') and not callable(getattr(mj_struct, attr))]
    return namedtuple(mj_struct.__class__.__name__, attrs)(**{attr: getattr(mj_struct, attr) for attr in attrs})