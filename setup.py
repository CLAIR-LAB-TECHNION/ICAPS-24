from setuptools import setup, find_packages
import glob
import os

motion_planning_files = glob.glob('motion_planning/ur5_rob/**/*', recursive=True)
motion_planning_files = [f.replace('motion_planning/', '') for f in motion_planning_files]
motion_planning_files.extend(['klampt_world.xml', "ur5.urdf"])

setup(
    name='ICAPS-24',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'GymJoCo @ git+https://github.com/guyazran/GymJoCo',
        'Klampt>=0.9.1',
        'mujoco>=3.1.3',
        'PyYAML>=6.0.1',
        'aidm[pddl] @ git+https://github.com/CLAIR-LAB-TECHNION/aidm',
        'mediapy'
    ],
    package_data={
        # 'mujoco_env': ["assets/scenes/3tableblocksworld/scene.xml"], # worked
        'motion_planning': motion_planning_files,
    }
)