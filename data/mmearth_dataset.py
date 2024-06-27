import json
from argparse import Namespace
from collections import OrderedDict
from copy import copy
from pathlib import Path
from typing import Union, Tuple

import ffcv
import h5py
import numpy as np
from ffcv import DatasetWriter
from ffcv.fields import NDArrayField, IntField, FloatField
from ffcv.fields.basics import IntDecoder
from ffcv.fields.ndarray import NDArrayDecoder
from ffcv.loader import OrderOption
from ffcv.transforms import ToTensor, Squeeze
from lightly.utils.dist import print_rank_zero
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from methods.transforms import to_tensor
from .constants import NO_DATA_VAL, MODALITIES_FULL, MODALITY_TASK, ori_input_size


##################### FUNCTIONS FOR PRETRAINING DATASETS #####################


class MMEarthDataset(Dataset):
    def __init__(self, args, split: str, transform = None, return_tuple: bool = False):
        # return_dict transform
        self.transform = transform

        # path to the dataset
        self.data_path = args.data_path
        # name of the dataset. for example: data_100k_130
        self.data_name = args.data_name
        # path to the split file
        self.splits_path = args.splits_path
        # tile info
        self.tile_info = args.tile_info
        # modalities used for training
        self.modalities = args.modalities
        # all modalities present in the datasets. This is used to keep track of the indices of the modalities in the dataset.
        self.modalities_full = args.modalities_full
        with open(self.splits_path, "r") as f:
            self.indices = json.load(f)[split]

        # mean, std, min and max of each band
        self.norm_stats = args.band_stats

        self.return_tuple = return_tuple

    def apply_transform(self, return_dict: dict):
        # TODO if more modalities are used, this will create a distorted view
        #  (e.g., flip applied to one but not to another modality)
        # applying transform for every pixel-wise modality
        for modality in return_dict:
            # only pixel based modalities
            if modality in [
                "sentinel2",
                "sentinel1",
                "aster",
                "canopy_height_eth",
                "dynamic_world",
                "esa_worldcover",
            ]:
                return_dict[modality] = self.transform(return_dict[modality])
        return return_dict

    def _open_hdf5(self, path: [str, Path]):
        self.data_full = h5py.File(path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):

        # this is to ensure that multiple workers do not open the same file multiple times.
        if not hasattr(self, "data_full"):
            self._open_hdf5(self.data_path)

        # based on what bands and what modalities we need for training, we return the return_dict[idx].)
        return_dict = OrderedDict()
        name = self.data_full["metadata"][self.indices[idx]][0].decode("utf-8")
        l2a = self.tile_info[name]["S2_type"] == "l2a"

        for modality in self.modalities.keys():
            # get the indices based on how it is in modalities_full
            if self.modalities[modality] == "all":
                modality_idx = [i for i in range(len(self.modalities_full[modality]))]
            else:
                modality_idx = [
                    self.modalities_full[modality].index(m)
                    for m in self.modalities[modality]
                ]

            if modality in ["biome", "eco_region"]:
                # for these modalities the array is already one hot encoded. hence modality_idx is not needed.
                data = np.argmax(self.data_full[modality][self.indices[idx]])
            else:
                # get the data
                # check if modalities are in ascending order, if not, swap for reading and swap for storing
                modality_idx = np.array(modality_idx)
                if np.all(np.diff(modality_idx) >= 0):
                    data = self.data_full[modality][self.indices[idx], modality_idx]
                else:
                    swap_idx = np.argsort(modality_idx)
                    data = self.data_full[modality][
                        self.indices[idx], modality_idx[swap_idx]
                    ][swap_idx]

            if modality == "dynamic_world":
                # the labels of dynamic world are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. We convert them to 0, 1, 2, 3, 4, 5, 6, 7, 8, nan respectively.
                # originally when downloading the no return_dict values are 0. hence we remap them to nan.
                data = np.where(data == NO_DATA_VAL[modality], np.nan, data)
                old_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]
                new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, np.nan]
                for old, new in zip(old_values, new_values):
                    data = np.where(data == old, new, data)
                # for any value greater than 8, we map them to nan
                data = np.where(data > 8, np.nan, data)

            if modality == "esa_worldcover":
                # the labels of esa worldcover are 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255.
                # We convert them to 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 255 respectively.
                old_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255]
                new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255]
                for old, new in zip(old_values, new_values):
                    data = np.where(data == old, new, data)
                # for any value greater than 10, we map them to nan
                data = np.where(data > 10, np.nan, data)

            # normalize
            if modality not in [
                "biome",
                "eco_region",
                "dynamic_world",
                "esa_worldcover",
            ]:
                if self.modalities[modality] == "all":
                    modality_idx = [
                        i for i in range(len(self.modalities_full[modality]))
                    ]
                else:
                    modality_idx = [
                        self.modalities_full[modality].index(m)
                        for m in self.modalities[modality]
                    ]

                # inside the band_stats, the name for sentinel2 is sentinel2_l1c or sentinel2_l2a
                if modality == "sentinel2":
                    modality_ = "sentinel2_l2a" if l2a else "sentinel2_l1c"
                else:
                    modality_ = modality

                means = np.array(self.norm_stats[modality_]["mean"])[modality_idx]
                stds = np.array(self.norm_stats[modality_]["std"])[modality_idx]
                if modality in ["era5", "lat", "lon", "month"]:
                    # single value mean and std
                    data = (data - means) / stds
                else:
                    # single value mean and std for each band
                    data = (data - means[:, None, None]) / stds[:, None, None]

            # converting the nodata values to nan to keep everything consistent
            data = (
                np.where(data == NO_DATA_VAL[modality], np.nan, data)
                if modality != "dynamic_world"
                else data
            )

            if MODALITY_TASK[modality] in ["classification", "segmentation"]:
                data = data.astype(np.dtype("int64"))
            else:
                data = data.astype(np.dtype("float32"))

            return_dict[modality] = data

        # we also return the id, to differentiate between sentinel2_l1c and sentinel2_l2a, since this is given in the tile_info json file. To keep everything
        # consistent, we name the modality as sentinel2 instead of sentinel2_l1c or sentinel2_l2a
        return_dict["id"] = name

        # apply transforms on normalized data
        if self.transform is not None:
            return_dict = self.apply_transform(return_dict)

        if self.return_tuple:
            return tuple(return_dict.values())

        return return_dict


def get_single_glob_file(data_root: Path, pattern) -> Path:
    file = [f for f in data_root.glob(pattern)]
    assert len(file) < 2, f"too many {pattern} files at {data_root}"
    assert len(file) > 0, f"no {pattern} files at {data_root}"
    return file[0]


def create_MMEearth_args(
    data_root: Path, input_modality: dict, target_modality: dict
) -> Namespace:
    args = Namespace()

    args.data_path = get_single_glob_file(data_root, "data_*.h5")
    args.splits_path = get_single_glob_file(data_root, "data_*_splits.json")
    args.tile_info_path = get_single_glob_file(data_root, "data_*_tile_info.json")
    with open(args.tile_info_path, "r") as f:
        args.tile_info = json.load(f)
    args.band_stats_path = get_single_glob_file(data_root, "data_*_band_stats.json")
    with open(args.band_stats_path, "r") as f:
        args.band_stats = json.load(f)
    args.data_name = data_root.name
    modalities = copy(input_modality)
    if target_modality is not None:
        modalities.update(target_modality)
    args.modalities = modalities
    args.modalities_full = MODALITIES_FULL
    return args


def get_mmearth_dataloaders(
    data_dir: Path,
    processed_dir: Path,
    input_modality: dict,
    target_modality: dict,
    num_workers: int,
    batch_size_per_device: int,
    splits: list[str] = None,
    no_ffcv: bool = False,
    indices: list[list[int]] = None,
) -> list[Union[ffcv.Loader, DataLoader]]:
    """
    Creates and returns data loaders for the MMEarth dataset. If the processed beton file does not exist, it processes the data
    and creates the beton file, then returns FFCV data loaders.

    Parameters:
    ----------
    data_dir : Path
        The directory where the raw dataset is stored.
    processed_dir : Path
        The directory where the processed beton files will be saved.
    input_modality : dict
        A dictionary specifying the input modality configurations.
    target_modality : dict
        A dictionary specifying the target modality configurations.
    num_workers : int
        The number of worker threads to use for data loading.
    batch_size_per_device : int
        The batch size for each device during training.
    splits : list[str], optional
        The dataset splits to be used. Default is ["train", "val"].
    no_ffcv: bool, optional
        Disables the creation of beton file and return torch Dataloader instead. Default is False.
    indices: list[list[int]], optional
        Select indices to use for each split (starting at 0). Default is None, meaning all samples are used. Only with FFCV enabled.

    Returns:
    -------
    list[Union[ffcv.Loader, torch.utils.data.DataLoader]]
        A list containing data loaders. Each loader can be either `ffcv.Loader` (for beton files) or `torch.data.DataLoader` (for standard PyTorch datasets).


    Example Usage:
    --------------
    ```python
    from pathlib import Path

    data_dir = Path("/path/to/raw/data")
    processed_dir = Path("/path/to/processed/data")
    input_modality = {...}  # Define your input modality configurations
    target_modality = {...}  # Define your target modality configurations
    num_workers = 4
    batch_size_per_device = 32

    dataloaders = get_mmearth_dataloaders(
        data_dir,
        processed_dir,
        input_modality,
        target_modality,
        num_workers,
        batch_size_per_device,
        splits=["train", "val"]
    )
    ```

    Notes:
    -----
    - The function checks if the processed beton file exists for each split. If it doesn't exist, it processes the data
      and creates the beton file.
    - The input and target modalities are reverse looked up using `IN_MODALITIES` and `MODALITIES_FULL` respectively.
    - The `convert_mmearth` function is used to convert the dataset into beton format.
    - The `ffcv.Loader` is used to create the data loaders with appropriate pipelines for training and validation.

    """
    if splits is None:
        splits = ["train", "val"]
    assert not no_ffcv or (
        no_ffcv and indices is None
    ), "Providing indices is not supported in no_ffcv mode."
    assert indices is None or (len(indices) == len(splits)), (
        "If indices are given, the number of splits and number of list of indices"
        "must align (len(indices) != len(splits) = ({len(indices)} != {len(splits))}"
    )

    if processed_dir is None:
        processed_dir = data_dir
    else:
        processed_dir.mkdir(exist_ok=True)

    # lookup input modality
    # only one input modality at a time supported TODO
    input_name = list(input_modality.keys())[0].replace("_", "-")

    # reverse lookup target modality
    if target_modality is None:
        target_name = ""
    else:
        # only one task supported TODO
        target_name = list(target_modality.keys())[0].replace("_", "-")

    dataloaders = []
    for i, split in enumerate(splits):
        is_train = split == "train"
        subset = "" if indices is None else "_subset"
        beton_file = processed_dir / f"{split}_{input_name}_{target_name}{subset}.beton"

        if not beton_file.exists() or no_ffcv:
            if not no_ffcv:
                print_rank_zero(
                    f"Processed file {beton_file} does not exist, trying to create it now."
                )
                transform = None
            else:
                transform = to_tensor
            args = create_MMEearth_args(data_dir, input_modality, target_modality)
            dataset = MMEarthDataset(
                args, split=split, transform=transform, return_tuple=True
            )

            if len(dataset) == 0:
                assert not is_train, "training dataset has no samples"
                print_rank_zero(
                    f"No samples in evaluation split '{split}', skipping it"
                )
                dataloaders.append(None)
                continue

            if no_ffcv:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size_per_device,
                    shuffle=is_train,
                    num_workers=num_workers,
                    drop_last=is_train,
                    persistent_workers=num_workers > 0,
                )
                dataloaders.append(dataloader)
                continue
            else:
                input_shape = (
                    sum([len(input_modality[k]) for k in input_modality]),
                    ori_input_size,
                    ori_input_size,
                )
                idx = None if indices is None else indices[i]
                convert_mmearth_to_beton(
                    dataset,
                    beton_file,
                    num_workers=num_workers,
                    input_shape=input_shape,
                    indices=idx,
                )

        # Data decoding and augmentation
        # Pipeline for each data field
        pipelines = {
            "sentinel2": [NDArrayDecoder(), ToTensor()],
        }

        if target_modality is not None:
            pipelines.update(
                {
                    "label": [
                        IntDecoder(),
                        ToTensor(),
                        Squeeze([1]),
                    ],  # this will only work for classification TODO
                }
            )

        # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
        dataloader = ffcv.Loader(
            beton_file,
            batch_size=batch_size_per_device,
            num_workers=num_workers,
            order=OrderOption.QUASI_RANDOM if is_train else OrderOption.SEQUENTIAL,
            pipelines=pipelines,
            drop_last=is_train,
        )

        dataloaders.append(dataloader)

    return dataloaders


def convert_mmearth_to_beton(
    dataset: MMEarthDataset,
    write_path: Path,
    input_shape: Tuple[int, int, int] = (12, 128, 128),
    num_workers: int = -1,
    indices: list = None,
):
    """
    Converts a MMEarth dataset into a format optimized for a specified machine learning task and writes it to a specified path.

    Parameters:
    ----------
    dataset : MMEarthDataset
        The dataset to be converted and written. It should be compatible with the DatasetWriter's from_indexed_dataset method.
    write_path : Path
        The file path where the transformed dataset will be written.
    input_shape : Tuple[int, int, int], optional
        The shape of the input data, used to define the shape of the sentinel2 and label fields when applicable. Default is (12, 128, 128).
    num_workers : int, optional
        The number of worker threads to use for writing the dataset. A value of -1 indicates that the default number of workers should be used. Default is -1.
    indices : list, optional
        Indices to select from dataset, good for subset creation.

    Fields:
    ------
    sentinel2 : NDArrayField
        A field for storing Sentinel-2 data with a specified shape and data type float32.
    label : IntField or FloatField or NDArrayField
        A field for storing labels, the type of which depends on the supervised_task parameter:
            - IntField for classification.
            - NDArrayField(dtype=np.dtype("float32"), shape=(c)) for regression.
            - NDArrayField(dtype=np.dtype("int64"), shape=(c, input_shape[1], input_shape[2])) for segmentation.
            - NDArrayField(dtype=np.dtype("float32"), shape=(c, input_shape[1], input_shape[2])) for regression map.

    Process:
    -------
    1. Field Initialization:
        Initializes the fields dictionary with a sentinel2 field.
        Adds a label field to the fields dictionary based on the supervised_task.
    2. Dataset Writing:
        Creates a DatasetWriter instance with the specified write_path, fields, and num_workers.
        Writes the dataset using the from_indexed_dataset method of the DatasetWriter.

    Example Usage:
    --------------
    ```python
    from pathlib import Path
    from data.mmearth_dataset import MMEarthDataset

    # Assuming 'my_dataset' is a pre-existing dataset object
    my_dataset = MMEarthDataset(...)  # Replace with actual dataset initialization

    convert_mmearth(
        dataset=my_dataset,
        write_path=Path('/path/to/save/dataset'),
        input_shape=(12, 128, 128),
        num_workers=4
    )
    ```
    """

    fields = {
        # Tune options to optimize dataset size, throughput at train-time
        "sentinel2": NDArrayField(dtype=np.dtype("float32"), shape=input_shape),
    }

    # if more than 1 modality is selected, we expect the second one to be the supervised task
    if len(dataset.modalities) >= 2:
        # only supporting single task here, so check the task and prepare fiel accordingly
        assert (
            len(dataset.modalities) == 2
        ), f"only two modalities should be returned, got: {dataset.modalities.keys()}"
        target_name, targets = [
            (k, v) for k, v in dataset.modalities.items() if k != "sentinel2"
        ][0]
        c = len(targets)  # number targets
        supervised_task = MODALITY_TASK[target_name]

        if supervised_task == "classification":
            if c == 1:
                fields.update({"label": IntField()})
            else:
                fields.update(
                    {"label": NDArrayField(dtype=np.dtype("int64"), shape=(c,))}
                )

        elif supervised_task == "segmentation":
            fields.update(
                {
                    "label": NDArrayField(
                        dtype=np.dtype("int64"),
                        shape=(c, input_shape[1], input_shape[2]),
                    )
                }
            )
        elif supervised_task == "regression":
            if c == 1:
                fields.update({"label": FloatField()})
            else:
                fields.update(
                    {"label": NDArrayField(dtype=np.dtype("float32"), shape=(c,))}
                )
        elif supervised_task == "regression_map":
            fields.update(
                {
                    "label": NDArrayField(
                        dtype=np.dtype("float32"),
                        shape=(c, input_shape[1], input_shape[2]),
                    )
                }
            )

    # Pass a type for each data field
    writer = DatasetWriter(write_path, fields, num_workers=num_workers)

    # Write dataset
    writer.from_indexed_dataset(dataset, indices=indices)
