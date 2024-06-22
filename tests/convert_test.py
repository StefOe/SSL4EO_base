import shutil
from pathlib import Path

from data import constants
from data.constants import MMEARTH_DIR
from data.geobench_dataset import convert_mmearth_to_beton
from data.mmearth_dataset import MMEarthDataset, create_MMEearth_args


def test_mmearth_dataset():
    split = "train"
    modalities = constants.INP_MODALITIES
    target = "biome"

    args = create_MMEearth_args(
        MMEARTH_DIR, modalities, {target: constants.MODALITIES_FULL[target]}
    )

    dataset = MMEarthDataset(args, split=split, transform=None, return_tuple=True)

    test_out = Path("test_out")
    test_out.mkdir(exist_ok=True)
    write_path = test_out / "mmearth.beton"

    try:
        convert_mmearth_to_beton(dataset, write_path, indices=[i for i in range(10)])
    finally:
        # cleanup
        shutil.rmtree(test_out, ignore_errors=True)
