# Transformer Guided Coevolution

Tested on Ubuntu 20.04, using Python 3.10 in a Miniconda environment.

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
  
