# Transformer Guided Coevolution
[Coevolutionary Deep RL](https://ieeexplore.ieee.org/document/9308290) implementation for use in multiagent adversarial team games. 
Uses various methods to solve the 'team selection' problem, including [BERTeam](https://github.com/pranavraj575/BERTeam), a transformer-based approach.
Interfaces with Pettingzoo multi-agent environments, experiments are done on [Pyquaticus](https://github.com/mit-ll-trusted-autonomy/pyquaticus), a simulated Marine Capture-The-Flag game implemented in Pettingzoo.

## Configuration
Run on Ubuntu 20.04, using Python 3.10 in a Miniconda environment.
Tested both on a Linux machine, and on Windows WSL (with Ubuntu version 20.04).



## Installation
* Install packages: `sudo apt install cmake swig zlib1g-dev`
* clone repo
  ```bash
  git clone --recurse-submodules https://github.com/pranavraj575/coevolution
  ```
* Install [conda](https://docs.anaconda.com/miniconda/#quick-command-line-install)
  * create conda env
    ```bash
    conda create --name coevolution python=3.10
    ```
  * OR run in coevolution folder
    ```bash
    conda create -f environment.yml
    ```
 * install project (run in coevolution folder)
   ```bash
   pip3 install -e .
   ```
 * install example repos used (run in coevolution folder)
   ```bash
   pip3 install -e repos/*
   ```
   OR choose which repos to install
   ```bash
   pip3 install -e repos/pyquaticus
   ```
  
