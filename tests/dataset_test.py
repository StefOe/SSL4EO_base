import json
from argparse import Namespace
from pathlib import Path

import pytest
from torch import Tensor

from data import constants
from data.geobench_dataset import GeobenchDataset
from data.mmearth_dataset import MultimodalDataset


@pytest.fixture
def args():
    args = Namespace()
    data_root = Path("./datasets/data_1k")
    assert data_root.exists(), f"need data (in {data_root}) to test this"
    args.data_path = data_root / "data_1k.h5"
    args.splits_path = data_root / "data_1k_splits.json"
    args.tile_info_path = data_root / "data_1k_tile_info.json"
    with open(args.tile_info_path, "r") as f:
        args.tile_info = json.load(f)
    args.band_stats_path = data_root / "data_1k_band_stats.json"
    with open(args.band_stats_path, "r") as f:
        args.band_stats = json.load(f)
    args.data_name = data_root.name

    args.modalities_full = constants.MODALITIES_FULL
    return args


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize(
    "modalities",
    [constants.OUT_MODALITIES, constants.INP_MODALITIES, constants.RGB_MODALITIES],
)
def test_mmearth_dataset(args, split, modalities):
    args.modalities = modalities
    dataset = MultimodalDataset(args, split=split, transform=None)

    if split == "train":
        assert len(dataset) > 0, "Dataset should not be empty"
        data = dataset[0]
        assert "sentinel2" in data, "Dataset should contain 'sentinel2' key"
        s1_channel = 8
        s2_channel = 12
        if modalities == constants.OUT_MODALITIES:
            assert isinstance(
                data["sentinel1"], Tensor
            ), "'sentinel1' data should be a Tensor"
            assert (
                data["sentinel1"].shape[0] == s1_channel
            ), f"'sentinel1' data should have {s1_channel} channels"
        elif modalities == constants.RGB_MODALITIES:
            s2_channel = 3
        assert isinstance(
            data["sentinel2"], Tensor
        ), "'sentinel2' data should be a Tensor"
        assert data["sentinel2"].shape[0] == s2_channel, (
            f"'sentinel2' data should have {s2_channel} channels "
            f"for {modalities.__name__}"
        )

    # no tests for val/test currently


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize(
    "dataset_name",
    [
        "m-eurosat",
        "m-so2sat",
        "m-bigearthnet",
        "m-brick-kiln",
        "m-cashew-plant",
        "m-SA-crop-type",
    ],
)
def test_geobench_dataset(args, split, dataset_name):
    if dataset_name in ["m-eurosat", "m-so2sat", "m-bigearthnet", "m-brick-kiln"]:
        dataset = GeobenchDataset(
            dataset_name=dataset_name,
            split=split,
            transform=None,
            benchmark_name="classification",
        )
    elif dataset_name in ["m-cashew-plant", "m-SA-crop-type"]:
        dataset = GeobenchDataset(
            dataset_name=dataset_name,
            split=split,
            transform=None,
            benchmark_name="segmentation",
        )
    else:
        raise NotImplementedError

    assert len(dataset) > 0, f"Dataset '{dataset_name}' should not be empty"

    n_channel = dataset[0][0].shape[0]
    expected = 12
    if dataset_name == "m-brick-kiln":
        expected = 3
    assert (
        n_channel == expected
    ), f"Dataset '{dataset_name}' should have {expected} channels, found {n_channel}"
