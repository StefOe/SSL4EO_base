import numpy as np
import torch
from torch import Tensor
from torch.nn import ModuleList, Module


def to_tensor(array: np.ndarray):
    return torch.from_numpy(array)

class MultiViewTransform(Module):
    def __init__(self, view_transforms: list[Module]):
        super().__init__()
        self.view_transforms = ModuleList(view_transforms)

    @torch.jit.script_method
    def forward(self, images: Tensor) -> list[Tensor]:
        # Apply each transform to the input in parallel to create different views
        views = [view_transform(images) for view_transform in self.view_transforms]
        return views