import pytest

from data import constants
from data.constants import MMEARTH_DIR
from data.mmearth_dataset import MultimodalDataset, create_MMEearth_args
from methods import transforms
from torchvision import transforms as T

from methods.transforms import to_tensor

input_size = 112


@pytest.mark.parametrize("split", ["train"])
@pytest.mark.parametrize("modalities", [constants.INP_MODALITIES])
@pytest.mark.parametrize(
    "transform",
    [
        transforms.BarlowTwinsTransform(
            transforms.BarlowTwinsView1Transform(input_size=input_size),
            transforms.BarlowTwinsView2Transform(input_size=input_size),
        ),
        transforms.SimCLRTransform(input_size=input_size),
        transforms.BYOLTransform(
            transforms.BYOLView1Transform(input_size=input_size),
            transforms.BYOLView2Transform(input_size=input_size),
        ),
        transforms.VICRegTransform(input_size=input_size),
        transforms.MAETransform(input_size=input_size),
    ],
)
def test_augmentations(split, modalities, transform):
    args = create_MMEearth_args(MMEARTH_DIR, modalities, constants.MODALITIES_FULL)

    args.modalities = modalities
    transform = T.Compose([to_tensor, transform])
    dataset = MultimodalDataset(args, split=split, transform=transform)

    if split == "train":
        num_samples = 10
        for i, data in enumerate(dataset):
            if i >= num_samples:
                break