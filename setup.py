from distutils.core import setup
from setuptools import find_packages

setup(
    name='coevolution',
    version='6.9.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'matplotlib',
        'pettingzoo',
        'pathos',
        'torch',
        'stable-baselines3',
        'unstable-baselines3',
    ],
    license='Liscence to Krill',
)
