import shutil
from argparse import Namespace
from pathlib import Path

import pytest
import torch.cuda

from data import constants
from eval import geobench_clf_eval
from main import main, METHODS


@pytest.fixture
def args():
    args = Namespace()
    data_root = constants.MMEARTH_DIR
    assert data_root.exists(), f"need data (in {data_root}) to test this"
    args.data_dir = data_root
    args.processed_dir = None
    args.log_dir = Path("test_out")
    args.batch_size_per_device = 2
    args.epochs = 2
    args.num_workers = 2
    args.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    args.devices = 1
    args.precision = "16-mixed"
    args.compile_model = False
    args.methods = None
    args.backbone = "default"
    args.input_channel = "all"
    args.target = "biome"
    args.last_backbone_channel = None
    args.geobench_datasets = None
    args.geobench_partitions = ["default"]
    args.geobench_eval_method = "linear"
    args.enable_knn_eval = True
    args.enable_linear_eval = True
    args.enable_finetune_eval = True
    args.ckpt_path = None
    args.no_ffcv = True

    return args


@pytest.mark.parametrize("methods", [k for k in METHODS])
@pytest.mark.parametrize(
    "geobench_datasets", [["m-eurosat"], ["m-so2sat"], ["m-bigearthnet"]]
)
def test_geobench_with_methods(args, methods: str, geobench_datasets: list[str]):
    args.log_dir.mkdir(exist_ok=True)
    args.methods = [methods]
    args.target = None
    args.geobench_datasets = geobench_datasets
    args.epochs = 0

    try:
        main(**vars(args), debug=True)
    finally:
        # cleanup
        shutil.rmtree(args.log_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "geobench_dataset", ["m-eurosat", "m-so2sat", "m-bigearthnet"]
)
def test_checkpointing_geobench(args, geobench_dataset: str):
    args.log_dir.mkdir(exist_ok=True)
    args.methods = ["vicreg"]
    args.backbone = "resnet18"
    args.target = None
    args.epochs = 0

    try:
        model = main(**vars(args), debug=True)

        geobench_clf_eval(
            model=model,
            method="linear",
            dataset_name=geobench_dataset,
            partition="default",
            log_dir=args.log_dir,
            processed_dir=args.log_dir,
            batch_size_per_device=args.batch_size_per_device,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            no_ffcv=args.no_ffcv,
            debug="long",
        )

    finally:
        # cleanup
        shutil.rmtree(args.log_dir, ignore_errors=True)


@pytest.mark.parametrize("methods", [k for k in METHODS])
@pytest.mark.parametrize("target", ["biome", "eco_region", None])
@pytest.mark.parametrize("last_backbone_channel", [None, 128])
def test_methods(args, methods: str, target, last_backbone_channel):
    args.log_dir.mkdir(exist_ok=True)
    if methods == "mae" and last_backbone_channel is not None:
        return  # this is not supported so skip
    args.methods = [methods]
    args.target = target
    args.last_backbone_channel = last_backbone_channel

    try:
        main(**vars(args), debug=True)
    finally:
        # cleanup
        shutil.rmtree(args.log_dir, ignore_errors=True)


