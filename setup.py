from setuptools import setup, find_packages
import glob
import os

#
# mujoco_package_data_files = ["mujoco_env/assets/scenes/3tableblocksworld/scene.xml",
#                              "mujoco_env/assets/robots/ur5e/robot.xml",
#                              "mujoco_env/assets/mounts/rethink_stationary/mount.xml",
#                              ]

setup(
    name='ICAPS-24',
    version='0.1.0',
    packages=['mujoco_env', 'motion_planning', 'n_table_blocks_world'],
    package_dir={'mujo_env': 'mujoco_env',
                 'motion_planning': 'motion_planning',
                 'n_table_blocks_world': 'n_table_blocks_world'},
    python_requires='>=3.9',
    install_requires=[
        'dm_control>=1.0.16',
        'gymnasium>=0.29.1',
        'Klampt>=0.9.1',
        'mujoco>=3.1.3',
        'PyYAML>=6.0.1',
    ],
    package_data={
        'mujoco_env': ["*.xml", "assets/scenes/3tableblocksworld/scene.xml"],
    }
)