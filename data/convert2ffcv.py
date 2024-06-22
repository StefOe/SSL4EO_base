from pathlib import Path
from typing import Tuple

import numpy as np
from ffcv.fields import NDArrayField, IntField, FloatField
from ffcv.writer import DatasetWriter


def convert_mmearth(
    dataset,
    write_path: Path,
    supervised_task: [str, None],
    input_shape: Tuple[int] = (12, 128, 128),
    num_workers: int = -1
):
    """
    Converts a dataset into a format optimized for a specified machine learning task and writes it to a specified path.

    Parameters:
    dataset : Any
        The dataset to be converted and written. It should be compatible with the DatasetWriter's from_indexed_dataset method.
    write_path : Path
        The file path where the transformed dataset will be written.
    supervised_task : str or None
        Specifies the type of supervised learning task. Options are:
            - "classification": For classification tasks.
            - "regression": For regression tasks.
            - "segmentation": For segmentation tasks.
            - "regression_map": For regression map tasks.
            - None: If no specific task type is provided.
    input_shape : Tuple[int], default=(12, 128, 128)
        The shape of the input data, used to define the shape of the sentinel2 and label fields when applicable.
    num_workers : int, default=-1
        The number of worker threads to use for writing the dataset. A value of -1 indicates that the default number of workers should be used.

    Fields:
    sentinel2 : NDArrayField
        A field for storing Sentinel-2 data with a specified shape and data type float32.
    label : IntField or FloatField or NDArrayField
        A field for storing labels, the type of which depends on the supervised_task parameter:
            - IntField for classification.
            - FloatField for regression.
            - NDArrayField(dtype=np.dtype("int64"), shape=input_shape) for segmentation.
            - NDArrayField(dtype=np.dtype("float32"), shape=input_shape) for regression map.

    Process:
    1. Field Initialization:
        Initializes the fields dictionary with a sentinel2 field.
        Adds a label field to the fields dictionary based on the supervised_task.
    2. Dataset Writing:
        Creates a DatasetWriter instance with the specified write_path, fields, and num_workers.
        Writes the dataset using the from_indexed_dataset method of the DatasetWriter.

    Example Usage:
    ```python
    from pathlib import Path

    # Assuming 'my_dataset' is a pre-existing dataset object
    convert_mmearth(
        dataset=my_dataset,
        write_path=Path('/path/to/save/dataset'),
        supervised_task='classification',
        input_shape=(12, 128, 128),
        num_workers=4
    )
    ```
    """

    fields = {
        # Tune options to optimize dataset size, throughput at train-time
        "sentinel2": NDArrayField(dtype=np.dtype("float32"), shape=input_shape),
    }
    if supervised_task == "classification":
        fields.update({"label": IntField()})
    elif supervised_task == "regression":
        fields.update({"label": FloatField()})
    elif supervised_task == "segmentation":
        fields.update({"label": NDArrayField(dtype=np.dtype("int64"), shape=input_shape)})
    elif supervised_task == "regression_map":
        fields.update({"label": NDArrayField(dtype=np.dtype("float32"), shape=input_shape)})

    # Pass a type for each data field
    writer = DatasetWriter(write_path, fields, num_workers=num_workers)

    # Write dataset
    writer.from_indexed_dataset(dataset)
