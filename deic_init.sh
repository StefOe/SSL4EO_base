#!/usr/bin/env bash

sudo apt-get update && sudo apt-get install -y libgl1-mesa-dev
eval "$(conda shell.bash hook)"
# Check if dir /work/cenv/ssl4eo exists; create conda env if not
if [ -d "/work/project/cenv/ssl4eo" ]; then
    export CONDA_ENVS_DIRS="/work/project/cenv/"
else
    echo "Creating conda environment"
    mamba env create --prefix /work/project/cenv/ssl4eo -f /work/data/env.yml
    export CONDA_ENVS_DIRS="/work/project/cenv/"
    # Set environment variables ensuring that env vars are always set when activating env
    mamba env config vars set MMEARTH_DIR=/work/data/MMEARTH100K/ GEO_BENCH_DIR=/work/data/geobench -n ssl4eo

fi
# Export new python kernel
/work/project/cenv/ssl4eo/bin/python -m ipykernel install --user --name ipy39 --display-name "SSL4EO"

# Set environment variables
export MMEARTH_DIR=/work/data/MMEARTH100K/
export GEO_BENCH_DIR=/work/data/geobench