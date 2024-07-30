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
        'torch==2.3.1',# TODO: test which versions of torch work with pathos
        'stable-baselines3',
        'unstable-baselines3',
    ],
    license='Liscence to Krill',
)
