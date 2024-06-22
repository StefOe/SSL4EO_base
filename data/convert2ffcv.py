from pathlib import Path
from typing import Tuple

import numpy as np
from ffcv.fields import NDArrayField, IntField, FloatField
from ffcv.writer import DatasetWriter

from .constants import MODALITY_TASK
from .mmearth_dataset import MMEarthDataset


def convert_mmearth(
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
