import json
from pathlib import Path
from typing import Tuple, Union

import ffcv
import geobench
import numpy as np
import torch
from ffcv import DatasetWriter
from ffcv.fields import NDArrayField, IntField, FloatField
from ffcv.fields.basics import IntDecoder
from ffcv.fields.ndarray import NDArrayDecoder
from ffcv.loader import OrderOption
from ffcv.transforms import ToTensor, Squeeze
from lightly.utils.dist import print_rank_zero
from torch.utils.data import Dataset, DataLoader

from data.constants import MODALITY_TASK, ori_input_size
from methods.transforms import to_tensor

with open("data/BAND_NAMES.json", "r") as f:
    BAND_NAMES = json.load(f)


class GeobenchDataset(Dataset):
    """paper introducing Geobench: https://arxiv.org/abs/2306.03831"""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform=None,
        partition: str = "default",
        benchmark_name: str = "classification",
    ):
        if split == "val":
            split = "valid"

        if benchmark_name == "classification":
            benchmark_name = "classification_v1.0/"
        elif benchmark_name == "segmentation":
            benchmark_name = "segmentation_v1.0/"

        task = None
        for task_ in geobench.task_iterator(benchmark_name=benchmark_name):
            if task_.dataset_name == dataset_name:
                task = task_
        assert task is not None, f"couldn't find {dataset_name} in {benchmark_name}"
        self.transform = transform
        self.dataset_name = dataset_name
        self.dataset = task.get_dataset(
            split=split,
            band_names=BAND_NAMES[dataset_name],
            partition_name=partition,
        )
        self.dataset_dir = task.get_dataset_dir()
        if hasattr(task.label_type, "class_names"):
            self.class_names = task.label_type.class_names
        elif hasattr(task.label_type, "class_name"):
            self.class_names = task.label_type.class_name
        else:
            raise Exception(f"Dataset {dataset_name} has no class names")
        self.num_classes = task.label_type.n_classes

        self.tmp_band_names = [band.name for band in task.bands_info]
        # get the tmp bands in the same order as the ones present in the BAND_NAMES.json file
        self.tmp_band_indices = [
            self.tmp_band_names.index(band_name)
            for band_name in BAND_NAMES[dataset_name]
        ]

        self.norm_stats = self.dataset.normalization_stats()
        self.in_channels = len(self.tmp_band_indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        label = self.dataset[idx].label
        x = []

        for band_idx in self.tmp_band_indices:
            x.append(self.dataset[idx].bands[band_idx].data)

        x = np.stack(x, axis=0)

        mean = np.array(self.norm_stats[0])
        std = np.array(self.norm_stats[1])

        if self.dataset_name == "m-so2sat":
            # the mean and std are multiplied by 10000 only for the so2sat dataset, while the
            # data values are in decimal range between 0 and 1. Hence, we need to divide the mean and std by 10000
            mean = mean / 10000
            std = std / 10000

        # normalize each band with its mean and std
        x = (x - mean[:, None, None]) / std[:, None, None]
        x = torch.from_numpy(x).float()

        # check if label is an object or a number
        if not (isinstance(label, int) or isinstance(label, list)):
            label = label.data
            # label is a memoryview object, convert it to a list, and then to a numpy array
            label = np.array(list(label))

        if self.transform is not None:
            self.transform(x)

        return x, label, mean, std


def get_geobench_dataloaders(
    dataset_name: str,
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
    Creates and returns data loaders for the GeobenchDataset dataset. If the processed beton file does not exist,
    it processes the data and creates the beton file, then returns FFCV data loaders.

    Parameters:
    ----------
    dataset_name : str
        Dataset name from geobench.
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
        The dataset splits to be used. Default is ["train", "val", "test"].
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

    dataloaders = get_geobench_dataloaders(
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
    - The `convert_geobench` function is used to convert the dataset into beton format.
    - The `ffcv.Loader` is used to create the data loaders with appropriate pipelines for training and validation.

    """
    if splits is None:
        splits = ["train", "val", "test"]
    assert not no_ffcv or (
        no_ffcv and indices is None
    ), "Providing indices is not supported in no_ffcv mode."
    assert indices is None or (len(indices) == len(splits)), (
        "If indices are given, the number of splits and number of list of indices"
        "must align (len(indices) != len(splits) = ({len(indices)} != {len(splits))}"
    )

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
            dataset = GeobenchDataset(
                dataset_name=dataset_name, split=split, transform=transform, partition=partition,
                benchmark_name=benchmark_name
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
                convert_geobench_to_beton(
                    dataset,
                    beton_file,
                    num_workers=num_workers,
                    input_shape=input_shape,
                    indices=idx,
                )

        # Data decoding and augmentation
        # Pipeline for each data field
        pipelines = {
            "input": [NDArrayDecoder(), ToTensor()],
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


def convert_geobench_to_beton(
    dataset: GeobenchDataset,
    write_path: Path,
    input_shape: Tuple[int, int, int],
    num_workers: int = -1,
    indices: list = None,
):
    """
    Converts a GeobenchDataset dataset into a format optimized for a specified machine learning task and writes it to a specified path.

    Parameters:
    ----------
    dataset : GeobenchDataset
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
    input : NDArrayField
        A field for storing input data with a specified shape and data type float32.
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
    from data.geobench_dataset import GeobenchDataset

    # Assuming 'my_dataset' is a pre-existing dataset object
    my_dataset = GeobenchDataset(...)  # Replace with actual dataset initialization

    convert_geobench(
        dataset=my_dataset,
        write_path=Path('/path/to/save/dataset'),
        input_shape=(12, 128, 128),
        num_workers=4
    )
    ```
    """

    fields = {
        # Tune options to optimize dataset size, throughput at train-time
        "input": NDArrayField(dtype=np.dtype("float32"), shape=input_shape),
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
