import torch
from ffcv.writer import DatasetWriter
from ffcv.fields import TorchTensorField, IntField

def convert_mmearth(dataset):
    # Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
    write_path = 'data/mmearth.beton'

    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'sentinel2': TorchTensorField(dtype=torch.float32, shape=(12, 128, 128)),
        'biome': IntField(),
        # 'id': BytesField(),
    }, num_workers=2)

    # Write dataset
    writer.from_indexed_dataset(dataset, shuffle_indices=True)