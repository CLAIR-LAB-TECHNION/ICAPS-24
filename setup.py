from setuptools import setup, find_packages
import glob
import os


mujoco_package_data_files = ["mujoco_env/assets/scenes/3tableblocksworld/scene.xml",
                             "mujoco_env/assets/robots/ur5e/robot.xml",
                             "mujoco_env/assets/mounts/rethink_stationary/mount.xml",
                             ]

setup(
    name='ICAPS-24',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'dm_control>=1.0.16',
        'gymnasium>=0.29.1',
        'Klampt>=0.9.1',
        'mujoco>=3.1.3',
        'numpy>=1.26.4',
        'PyYAML>=6.0.1',
        'scipy>=1.13.0'
    ],
    package_data={
        'mujoco_env': mujoco_package_data_files
    }
)