import shutil
from pathlib import Path

import numpy as np
import pytest

from data import GeobenchDataset, get_mmearth_dataloaders
from data import MMEarthDataset, create_MMEearth_args
from data import constants


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize(
    "modalities",
    [constants.INP_MODALITIES, constants.RGB_MODALITIES],
)
@pytest.mark.parametrize(
    "target_modalities",
    [
        {"biome": constants.MODALITIES_FULL["biome"]},
        {"eco_region": constants.MODALITIES_FULL["eco_region"]},
    ],
)
def test_mmearth_dataset(split, modalities, target_modalities):
    args = create_MMEearth_args(constants.MMEARTH_DIR, modalities, target_modalities)

    dataset = MMEarthDataset(args, split=split, transform=None)

    if split == "train":
        assert len(dataset) > 0, "Dataset should not be empty"
        data = dataset[0]
        assert "sentinel2" in data, "Dataset should contain 'sentinel2' key"
        s1_channel = 8
        s2_channel = 12
        if modalities == constants.OUT_MODALITIES:
            assert isinstance(
                data["sentinel1"], np.ndarray
            ), "'sentinel1' data should be a Tensor"
            assert (
                data["sentinel1"].shape[0] == s1_channel
            ), f"'sentinel1' data should have {s1_channel} channels"
        elif modalities == constants.RGB_MODALITIES:
            s2_channel = 3
        assert isinstance(
            data["sentinel2"], np.ndarray
        ), "'sentinel2' data should be a Tensor"
        assert (
            data["sentinel2"].shape[0] == s2_channel
        ), f"'sentinel2' data should have {s2_channel} channels"

    # no tests for val/test currently


@pytest.mark.parametrize(
    "modalities",
    [constants.INP_MODALITIES, constants.RGB_MODALITIES],
)
@pytest.mark.parametrize(
    "target_modalities",
    [
        {"biome": constants.MODALITIES_FULL["biome"]},
        {"eco_region": constants.MODALITIES_FULL["eco_region"]},
    ],
)
@pytest.mark.parametrize(
    "no_ffcv", [False, True],
)
def test_mmearth_dataloader(modalities, target_modalities, no_ffcv):
    test_out = Path("test_out")
    test_out.mkdir(exist_ok=True)

    try:
        loader = get_mmearth_dataloaders(
            None, constants.MMEARTH_DIR, test_out,
            modalities, target_modalities, 2, 2, ["train"], no_ffcv,
            indices=None if no_ffcv else [list(range(10))]
        )

        for data in loader:
            break
    finally:
        # cleanup
        shutil.rmtree(test_out, ignore_errors=True)



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
def test_geobench_dataset(split, dataset_name):
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
