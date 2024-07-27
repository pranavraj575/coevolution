# Title pending

## Installation
* Install packages: `sudo apt install cmake swig zlib1g-dev`
* clone repo
  ```bash
  git clone --recurse-submodules https://github.com/pranavraj575/coevolution
  ```
* Install [conda](https://docs.anaconda.com/miniconda/#quick-command-line-install)
  * create conda env
    ```bash
    conda create --name coevolution
    ```
  * OR run in coevolution folder
    ```bash
    conda create -f environment.yml
    ```
 * install project (run in coevolution folder)
   ```bash
   pip3 install -e .
   ```
