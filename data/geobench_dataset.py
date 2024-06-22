import json
from pathlib import Path
from typing import Tuple

import geobench
import numpy as np
import torch
from ffcv import DatasetWriter
from ffcv.fields import NDArrayField, IntField, FloatField
from torch.utils.data import Dataset

from data import MMEarthDataset
from data.constants import MODALITY_TASK

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
            (k, v) for k, v in dataset.modalities.item() if k != "sentinel2"
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
