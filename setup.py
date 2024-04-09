from setuptools import setup, find_packages
import glob
import os


# Recursively include all files in the mujoco_env/assets directory
package_data_files = glob.glob('mujoco_env/assets/**/*', recursive=True)
# Filter out directories from the list
package_data_files = [f for f in package_data_files if not os.path.isdir(f)]

setup(
    name='icaps24',
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
        'mujoco_env': package_data_files
    }
)