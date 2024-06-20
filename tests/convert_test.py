import pytest
from torch import Tensor

from data import constants
from data.constants import MMEARTH_DIR
from data.convert2ffcv import convert_mmearth
from data.geobench_dataset import GeobenchDataset
from data.mmearth_dataset import MultimodalDataset, create_MMEearth_args


@pytest.mark.parametrize("split", ["train"])
@pytest.mark.parametrize(
    "modalities",
    [constants.INP_MODALITIES],
)
def test_mmearth_dataset(split, modalities):
    args = create_MMEearth_args(
        MMEARTH_DIR, modalities, {"biome": constants.MODALITIES_FULL["biome"]}
    )

    dataset = MultimodalDataset(args, split=split, transform=None)

    convert_mmearth(dataset)
