import json
from pathlib import Path
from typing import Union, Tuple

import ffcv
import geobench
import numpy as np
from ffcv import DatasetWriter
from ffcv.fields import NDArrayField, IntField
from ffcv.fields.basics import IntDecoder
from ffcv.fields.ndarray import NDArrayDecoder
from ffcv.loader import OrderOption
from ffcv.transforms import ToTensor, Squeeze
from geobench import MultiLabelClassification, TaskSpecifications, SemanticSegmentation, SegmentationClasses
from lightly.utils.dist import print_rank_zero
from torch.utils.data import Dataset, DataLoader

from methods.transforms import to_tensor

with open("data/BAND_NAMES.json", "r") as f:
    BAND_NAMES = json.load(f)

GEOBENCH_TASK = {
    "m-eurosat": "classification",
    "m-so2sat": "classification",
    "m-bigearthnet": "classification",
    "m-brick-kiln": "classification",
    "m-cashew-plant": "segmentation",
    "m-SA-crop-type": "segmentation",
}


class GeobenchDataset(Dataset):
    """paper introducing Geobench: https://arxiv.org/abs/2306.03831"""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform=None,
        partition: str = "default",
    ):
        if split == "val":
            split = "valid"

        task, benchmark_name = self.get_task(dataset_name)

        self.benchmark_name = benchmark_name
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
        self.patch_size = task.patch_size
        self.label_type = task.label_type

        self.norm_stats = self.dataset.normalization_stats()
        self.in_channels = len(self.tmp_band_indices)

    @staticmethod
    def get_task(dataset_name: str) -> Tuple[TaskSpecifications, str]:
        benchmark_name = GEOBENCH_TASK[dataset_name]
        benchmark_name_ = ""
        if benchmark_name == "classification":
            benchmark_name_ = "classification_v1.0/"
        elif benchmark_name == "segmentation":
            benchmark_name_ = "segmentation_v1.0/"
        task = None
        for task_ in geobench.task_iterator(benchmark_name=benchmark_name_):
            if task_.dataset_name == dataset_name:
                task = task_
        assert task is not None, f"couldn't find {dataset_name} in {benchmark_name_}"
        return task, benchmark_name

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

        # cast to float32
        x = x.astype(np.dtype("float32"))
        mean = mean.astype(np.dtype("float32"))
        std = std.astype(np.dtype("float32"))

        # check if label is an object or a number
        if not (isinstance(label, int) or isinstance(label, list)):
            label = label.data
            # label is a memoryview object, convert it to a list, and then to a numpy array
            label = np.array(list(label), dtype=np.dtype("int64"))

        if self.transform is not None:
            self.transform(x)

        return x, label, mean, std


def get_geobench_dataloaders(
    dataset_name: str,
    processed_dir: Path,
    num_workers: int,
    batch_size_per_device: int,
    splits: list[str] = None,
    partition: str = "default",
    no_ffcv: bool = False,
    indices: list[list[int]] = None,
) -> Tuple[list[Union[ffcv.Loader, DataLoader]], TaskSpecifications]:
    """
    Creates and returns data loaders for the GeobenchDataset dataset. If the processed beton file does not exist,
    it processes the data and creates the beton file, then returns FFCV data loaders.

    Parameters:
    ----------
    dataset_name : str
        The name of the dataset from Geobench.
    processed_dir : Path
        The directory where the processed beton files will be saved.
    num_workers : int
        The number of worker threads to use for data loading.
    batch_size_per_device : int
        The batch size for each device during training.
    splits : list[str], optional
        The dataset splits to be used. Default is ["train", "val", "test"].
    partition : str, optional
        The partition strategy for the dataset. Default is "default".
    no_ffcv : bool, optional
        Disables the creation of beton files and returns PyTorch DataLoader instead. Default is False.
    indices : list[list[int]], optional
        Select indices to use for each split (starting at 0). Default is None, meaning all samples are used. Only applicable with FFCV enabled.

    Returns:
    -------
    Tuple[list[Union[ffcv.Loader, torch.utils.data.DataLoader]], TaskSpecifications]
        A tuple containing a list of data loaders and task specifications. Each loader can be either `ffcv.Loader` (for beton files) or `torch.utils.data.DataLoader` (for standard PyTorch datasets).

    Example Usage:
    --------------
    ```python
    from pathlib import Path

    data_dir = Path("/path/to/raw/data")
    processed_dir = Path("/path/to/processed/data")
    num_workers = 4
    batch_size_per_device = 32

    dataloaders = get_geobench_dataloaders(
        dataset_name="dataset_name",
        processed_dir=processed_dir,
        num_workers=num_workers,
        batch_size_per_device=batch_size_per_device,
        splits=["train", "val"]
    )
    ```

    Notes:
    -----
    - The function checks if the processed beton file exists for each split. If it doesn't exist, it processes the data
      and creates the beton file.
    - The `convert_geobench_to_beton` function is used to convert the dataset into beton format.
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

    dataloaders = []
    task, _ = GeobenchDataset.get_task(dataset_name)
    for i, split in enumerate(splits):
        is_train = split == "train"
        subset = "" if indices is None else "_subset"
        beton_file = processed_dir / f"{split}_{dataset_name}_{partition}{subset}.beton"

        if not beton_file.exists() or no_ffcv:
            if not no_ffcv:
                print_rank_zero(
                    f"Processed file {beton_file} does not exist, trying to create it now."
                )
                transform = None
            else:
                transform = to_tensor
            dataset = GeobenchDataset(
                dataset_name=dataset_name,
                split=split,
                transform=transform,
                partition=partition,
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
                idx = None if indices is None else indices[i]
                convert_geobench_to_beton(
                    dataset,
                    beton_file,
                    num_workers=num_workers,
                    indices=idx,
                )

        # Data decoding and augmentation
        # Pipeline for each data field
        pipelines = {
            "input": [NDArrayDecoder(), ToTensor()],
        }
        # get correct decoder for task
        if isinstance(task.label_type, (MultiLabelClassification, SemanticSegmentation, SegmentationClasses)):
            pipelines.update(
                {
                    "label": [
                        NDArrayDecoder(),
                        ToTensor(),
                    ],
                }
            )
        else:
            pipelines.update(
                {
                    "label": [
                        IntDecoder(),
                        ToTensor(),
                        Squeeze([1]),
                    ],
                }
            )
        pipelines.update(
            {
                "mean": [NDArrayDecoder(), ToTensor()],
                "std": [NDArrayDecoder(), ToTensor()],
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

    return dataloaders, task


def convert_geobench_to_beton(
    dataset: GeobenchDataset,
    write_path: Path,
    num_workers: int = -1,
    indices: list = None,
):
    """
    Converts a GeobenchDataset into a format optimized for a specified machine learning task and writes it to a specified path.

    Parameters:
    ----------
    dataset : GeobenchDataset
        The dataset to be converted and written. It should be compatible with the DatasetWriter's from_indexed_dataset method.
    write_path : Path
        The file path where the transformed dataset will be written.
    num_workers : int, optional
        The number of worker threads to use for writing the dataset. A value of -1 indicates that the default number of workers should be used. Default is -1.
    indices : list, optional
        Indices to select from the dataset, useful for subset creation.

    Fields:
    ------
    input : NDArrayField
        A field for storing input data with a specified shape and data type float32.
    label : IntField or FloatField or NDArrayField
        A field for storing labels, the type of which depends on the supervised_task parameter:
            - IntField for multi-class classification.
            - NDArrayField(dtype=np.dtype("int64"), shape=(c)) for multi-label classification.
            - NDArrayField(dtype=np.dtype("int64"), shape=(c, input_shape[1], input_shape[2])) for segmentation.

    Process:
    -------
    1. Field Initialization:
        Initializes the fields dictionary with an 'input' field.
        Adds a 'label' field to the fields dictionary based on the supervised_task.
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

    convert_geobench_to_beton(
        dataset=my_dataset,
        write_path=Path('/path/to/save/dataset'),
        num_workers=4
    )
    ```

    Notes:
    -----
    - The `convert_geobench_to_beton` function facilitates the conversion of a GeobenchDataset into a beton format optimized for machine learning tasks.
    - The function initializes appropriate fields based on the dataset type (classification, multi-label classification, segmentation) and writes the dataset to the specified path.
    - The `from_indexed_dataset` method of `DatasetWriter` is used to handle the actual writing process.
    """

    input_shape = (
        dataset.in_channels,
        *dataset.patch_size,
    )

    fields = {
        # Tune options to optimize dataset size, throughput at train-time
        "input": NDArrayField(dtype=np.dtype("float32"), shape=input_shape),
    }

    # check if this is a multi label dataset
    if isinstance(dataset.label_type, MultiLabelClassification):
        c = dataset.num_classes  # number target tasks
    else:
        c = 1  # in Multiclass classification only one output exists

    if dataset.benchmark_name == "classification":
        if c == 1:
            fields.update({"label": IntField()})
        else:
            fields.update({"label": NDArrayField(dtype=np.dtype("int64"), shape=(c,))})

    elif dataset.benchmark_name == "segmentation":
        fields.update(
            {
                "label": NDArrayField(
                    dtype=np.dtype("int64"),
                    shape=(c, input_shape[1], input_shape[2]),
                )
            }
        )

    fields.update(
        {
            "mean": NDArrayField(
                dtype=np.dtype("float32"),
                shape=(input_shape[0],),
            ),
            "std": NDArrayField(
                dtype=np.dtype("float32"),
                shape=(input_shape[0],),
            ),
        }
    )

    # Pass a type for each data field
    writer = DatasetWriter(write_path, fields, num_workers=num_workers)

    # Write dataset
    writer.from_indexed_dataset(dataset, indices=indices)
