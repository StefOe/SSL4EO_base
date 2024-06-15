import json

import numpy as np
import torch
from torch.utils.data import Dataset

##################### FUNCTIONS FOR FINE-TUNING DATASETS #####################

with open("data/BAND_NAMES.json", "r") as f:
    BAND_NAMES = json.load(f)


class GeobenchDataset(Dataset):
    def __init__(
        self,
        dataset_name=None,
        split="train",
        transform=None,
        benchmark_name="classification",
    ):
        if split == "val":
            split = "valid"

        if benchmark_name == "classification":
            benchmark_name = "classification_v0.9.1/"
        elif benchmark_name == "segmentation":
            benchmark_name = "segmentation_v0.9.1/"

        for task in geobench.task_iterator(benchmark_name=benchmark_name):
            if task.dataset_name == dataset_name:
                break
        self.transform = transform
        self.dataset_name = dataset_name
        self.dataset = task.get_dataset(
            split=split, band_names=BAND_NAMES[dataset_name]
        )
        self.label_map = task.get_label_map()
        self.label_stats = (
            task.label_stats() if benchmark_name != "segmentation_v0.9.1/" else "None"
        )
        self.dataset_dir = task.get_dataset_dir()
        if dataset_name == "m-brick-kiln":
            self.num_classes = 2
        elif dataset_name == "m-bigearthnet":
            self.num_classes = 43
        elif dataset_name == "m-cashew-plantation":
            self.num_classes = 7
        elif dataset_name == "m-SA-crop-type":
            self.num_classes = 10
        else:
            self.num_classes = len(task.get_label_map().keys())
        self.tmp_band_names = [
            self.dataset[0].bands[i].band_info.name
            for i in range(len(self.dataset[0].bands))
        ]
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

        return x, label, mean, std


class GeobenchDatasetSubset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.in_channels = dataset.in_channels
        self.num_classes = dataset.num_classes
        self.norm_stats = dataset.norm_stats
        self.dataset_name = dataset.dataset_name

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
