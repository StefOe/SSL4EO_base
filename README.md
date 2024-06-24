# SSL4EO

Code base for the course: Self-Supervised Learning for Earth Observation

## Setup

Conda or Mamba (preferred) is required for the setup.
We assume that you have a NVIDIA GPU available.

1. create the python env: `mamba env create -f env.yml`
2. activate the env: `mamba activate ssl4eo`
2. download the MMEarth data: TODO
3. set the env variable to your MMEarth directory: `export MMEARTH_DIR=<your path> `
4. to download geobench data, run: `geobench-download`
5. (optional) run the tests: `pytest`

## Start training

example for training with VICReg:
`python main --methods vicreg`