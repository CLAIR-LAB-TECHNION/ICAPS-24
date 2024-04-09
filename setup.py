from setuptools import setup, find_packages
import glob
import os


# all files inside assets dir, recursively
mujoco_env_files = glob.glob('mujoco_env/assets/**/*', recursive=True)
# remove mujoco_env prefix:
mujoco_env_files = [f.replace('mujoco_env/', '') for f in mujoco_env_files]

motion_planning_files = glob.glob('motion_planning/ur_description/**/*', recursive=True)
motion_planning_files = [f.replace('motion_planning/', '') for f in motion_planning_files]

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
        'PyYAML>=6.0.1',
    ],
    package_data={
        # 'mujoco_env': ["assets/scenes/3tableblocksworld/scene.xml"], # worked
        'mujoco_env': mujoco_env_files,
        'motion_planning': motion_planning_files,
    }
)