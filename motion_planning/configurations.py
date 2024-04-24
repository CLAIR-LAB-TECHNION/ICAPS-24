import numpy as np

# constraint joint limits for faster planning. joint 2 is base rotation, don't need entire 360,
# joint 3 is shoulder lift, outside the limits will probably result table collision anyway.
# joint 4 has 2 pi range anyway. all the others can do fine with 3pi range
limits_l = [0, - 3 * np.pi / 2, -4.5, -np.pi, -3 * np.pi / 2, -3 * np.pi / 2, -3 * np.pi / 2, 0]
limits_h = [0, np.pi / 2, 1., np.pi, 3 * np.pi / 2, 3 * np.pi / 2, 3 * np.pi / 2, 0]

default_config = {
    # "type": "lazyrrg*",
    "type": "rrt*",
    "bidirectional": True,
    "connectionThreshold": 30.0,
    # "shortcut": True, # only for rrt
}
