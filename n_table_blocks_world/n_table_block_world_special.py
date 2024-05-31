from copy import deepcopy
from n_table_blocks_world.n_table_blocks_world import NTableBlocksWorld
from n_table_blocks_world.configurations_and_constants import env_cfg


full_table_env_cfg = deepcopy(env_cfg)
full_table_env_cfg['scene']['resource'] = '3tableblocksworld_fulltable'

class NTableBlocksWorldFullTable(NTableBlocksWorld):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode, cfg=full_table_env_cfg)


start_near_table2_cfg = deepcopy(env_cfg)
start_near_table2_cfg['scene'].pop('init_keyframe')
start_near_table2_cfg['robot']['init_pos'] = [3.9539, -0.3656,  1.3086,  3.7935, -1.5708, -0.7423]

class NTableBlocksWorldStartNearTable2(NTableBlocksWorld):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode, cfg=start_near_table2_cfg)

