
<p align="right"><img src="ssl4eo-logo.png" width="150"/></p>

# SSL4EO  
Code base for the course: Self-Supervised Learning for Earth Observation

Here are the logs for running the model with default settings for 50 and 100 epochs: [wandb results pretraining](https://wandb.ai/stefoe/ssl4eo/?nw=nwuserstefoe)

## Local Setup
Conda or Mamba (preferred) is required for the setup.
We assume that you have a NVIDIA GPU available.

1. create the python env: `mamba env create -f env.yml`
2. activate the env: `mamba activate ssl4eo`
2. download the MMEarth data (~45 GB): `curl -L https://sid.erda.dk/share_redirect/fnCZOGsWDC -o data_100k_v001.zip`
3. Make a directory for data: `mkdir <your path>`
4. unzip the folder: `unzip data_100k_v001.zip -d <your path>`
5. set the env variable to your MMEarth directory: `mamba env config vars set -n ssl4eo MMEARTH_DIR=<your path>`
6. reload environment to ensure that env variable is set: `mamba activate ssl4eo`
7. to download geobench data, run: `geobench-download`
8. (optional) get pretrained weights: `curl -L https://sid.erda.dk/share_redirect/DGCdXRPvNg -o weights.zip`
9. (optional) unzip somewhere: `unzip weights.zip -d <path to somewhere>`
10. (optional) run the tests (takes some time): `pytest`

## Deic Setup

Follow the _[SSL4EO Mini-Projects instructions - Compute access on DEIC](https://docs.google.com/document/d/1E4yG7y6fgcgvodaDsTtrts-Aiw2nb32s8Tb3S-w4C38/edit?usp=sharing)_ to get started with DeiC. Once you have access to DeiC - course resources and started a container and run the following. It will install and prepare your conda env in ~10-25 mins: 

## Examples
The default setting for the pretraining is that all data is used and "biome" is used as the target for the online classifier. Also, if not specified, all methods are used.

Get an overview of commands:
`python main.py --help`

Pretraining with VICReg:
`python main.py --methods vicreg`

Evaluating on bigeartnet at the end of the pretraining with SimClr:
`python main.py --methods simclr --geobench-datasets=m-bigearthnet`

Evaluating on bigeartnet with pretrained barlowtwins model:
`python main.py --methods barlowtwins --geobench-datasets=m-bigearthnet --epochs=0  --ckpt-path=/work/data/weights/barlowtwins/50epochs.ckpt`

When changing the main dataset, you will need to recreate the optimized dataformat.
Therefore specify your processed folder to be a writeable directory. Here for an example when pretraining with "eco_region" (instead of biome) as online linear probing target (all methods):
`python main.py --target=eco_region --processed_dir=/work/project`

Another example that needs newly processed data, we only use 10% training data for bigearthnet:
`python main.py --methods barlowtwins --processed_dir=/work/project --geobench-datasets=m-bigearthnet --geobench-partition`

