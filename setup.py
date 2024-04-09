from setuptools import setup, find_packages
import glob
import os


# all files inside assets dir, recursively
mujoco_env_files = glob.glob('mujoco_env/assets/**/*', recursive=True)


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

    }
)