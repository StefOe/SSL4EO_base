import json
from argparse import Namespace
from copy import copy
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.constants import NO_DATA_VAL, MODALITIES_FULL


##################### FUNCTIONS FOR PRETRAINING DATASETS #####################


class MultimodalDataset(Dataset):
    def __init__(self, args, split: str, transform, return_tuple: bool = False):
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
        return_dict = {}
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
            if modality in [
                "biome",
                "eco_region",
                "dynamic_world",
                "esa_worldcover",
            ]:
                data = data.astype(int)
            else:
                data = data.astype(np.dtype("float32"))

            return_dict[modality] = data

        # we also return the id, to differentiate between sentinel2_l1c and sentinel2_l2a, since this is given in the tile_info json file. To keep everything
        # consistent, we name the modality as sentinel2 instead of sentinel2_l1c or sentinel2_l2a
        return_dict["id"] = np.array(name)

        # apply transforms on normalized data
        if self.transform is not None:
            return_dict = self.apply_transform(return_dict)

        if self.return_tuple:
            return tuple(return_dict.values())

        return return_dict


def get_single_glob_file(data_root: Path, pattern) -> Path:
    file = [f for f in data_root.glob(pattern)]
    assert len(file) == 1, f"too many {pattern} files at {data_root}"
    return file[0]

def create_MMEearth_args(data_root: Path, input_modality:dict, target_modality:dict) -> Namespace:
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
