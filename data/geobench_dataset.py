import json

import geobench
import numpy as np
import torch
from torch.utils.data import Dataset

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
