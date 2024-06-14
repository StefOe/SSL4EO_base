import json
from argparse import Namespace
from pathlib import Path
import pytest
from torch import Tensor

from data import MODALITIES
from data.mmearth_dataset import MultimodalDataset


@pytest.fixture
def args():
    args = Namespace()
    data_root = Path("./datasets/data_1k")
    args.data_path = data_root / "data_1k.h5"
    args.splits_path = data_root / "data_1k_splits.json"
    args.tile_info_path = data_root / "data_1k_tile_info.json"
    with open(args.tile_info_path, "r") as f:
        args.tile_info = json.load(f)
    args.band_stats_path = data_root / "data_1k_band_stats.json"
    with open(args.band_stats_path, 'r') as f:
        args.band_stats = json.load(f)
    args.data_name = data_root.name

    args.modalities = MODALITIES.OUT_MODALITIES
    args.modalities_full = MODALITIES.MODALITIES_FULL
    return args


@pytest.mark.parametrize("random_crop", [False, True])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_mmearth_dataset(args, random_crop, split):
    args.random_crop = random_crop

    if random_crop:
        args.input_size = 64

    dataset = MultimodalDataset(args, split=split)

    if split == "train":
        assert len(dataset) > 0, "Dataset should not be empty"
        data = dataset[0]
        assert 'sentinel2' in data, "Dataset should contain 'sentinel2' key"
        s2_channel = 12
        assert isinstance(data['sentinel2'], Tensor), "'sentinel2' data should be a Tensor"
        assert data['sentinel2'].shape[0] == s2_channel, f"'sentinel2' data should have {s2_channel} channels"
        s1_channel = 8
        assert isinstance(data['sentinel1'], Tensor), "'sentinel1' data should be a Tensor"
        assert data['sentinel1'].shape[0] == s1_channel, f"'sentinel1' data should have {s1_channel} channels"

        if random_crop:
            assert data['sentinel2'].shape[1] == args.input_size, "input should be multicropped"
            assert data['sentinel2'].shape[2] == args.input_size, "input should be multicropped"
    else:
        assert len(dataset) == 0, f"{split} dataset should be empty"
