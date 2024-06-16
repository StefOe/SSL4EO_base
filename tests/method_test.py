import shutil
from argparse import Namespace
from pathlib import Path

import pytest
import torch.cuda

from main import main


@pytest.fixture
def args():
    args = Namespace()
    data_root = Path("./datasets/data_1k")
    assert data_root.exists(), f"need data (in {data_root}) to test this"
    args.data_dir = data_root
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
    args.skip_knn_eval = False
    args.skip_linear_eval = False
    args.skip_finetune_eval = False
    args.ckpt_path = None

    return args


@pytest.mark.parametrize("target", ["biome", "eco_region", None])
@pytest.mark.parametrize("last_backbone_channel", [None, 128])
def test_methods(args, target, last_backbone_channel):
    args.log_dir.mkdir(exist_ok=True)
    args.target = target
    args.last_backbone_channel = last_backbone_channel
    if target is None:
        args.skip_knn_eval = args.skip_linear_eval = args.skip_finetune_eval = True

    try:
        main(**vars(args), debug=True)
    finally:
        # cleanup
        shutil.rmtree(args.log_dir, ignore_errors=True)
