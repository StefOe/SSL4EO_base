#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
# Check if dir /work/cenv/ssl4eo exists; create conda env if not
if [ -d "/work/project/cenv/ssl4eo" ]; then
    export CONDA_ENVS_DIRS="/work/project/cenv/"
else
    echo "Creating conda environment"
    mamba env create --prefix /work/project/cenv/ssl4eo -f /work/data/env.yml
    export CONDA_ENVS_DIRS="/work/project/cenv/"
fi
# Export new python kernel
python -m ipykernel install --user --name ipy39 --display-name "SSL4EO"

# Set environment variables
export MMEARTH_DIR=/work/data/MMEARTH100K/
export GEO_BENCH_DIR=/work/data/geobench