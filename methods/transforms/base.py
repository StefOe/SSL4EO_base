from dataclasses import replace
from typing import Callable, Tuple, Optional

import numpy as np
import torch
from ffcv.pipeline import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from lightly.transforms.multi_view_transform import MultiViewTransform
from numpy.random import rand
from torch import Tensor
from torchvision.transforms import Compose


def to_tensor(array:np.ndarray):
    return torch.from_numpy(array)


class MultiViewOperation(MultiViewTransform, Operation):
    input_size = None
    def generate_code(self) -> Callable:
        def transform(image_batch: Tensor, _): # input dim: b x c x h x w -> output sim: v x b x c x h' x w'
            # apply transform to each image individually
            views = []
            for transform in self.transforms:
                batch = []
                for image in image_batch:
                    batch.append(transform(image))
                views.append(torch.stack(batch, dim=0))
            return views

            # this is naively apply same augmentation to all images -> faster but less data variability
            # return self.__call__(image_batch) #
        return transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        shape = (previous_state.shape[0], self.input_size, self.input_size)
        return (
            replace(
                previous_state,
                shape=shape,
                # shape=(len(self.transforms), previous_state.shape[0], self.input_size, self.input_size),
            ),
            AllocationQuery(shape, previous_state.dtype),
        )

class FFCVCompose(Compose, Operation):
    def generate_code(self) -> Callable:
        def transform(images, _):
            return self.__call__(images)

        return transform

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (previous_state,  AllocationQuery(previous_state.shape, previous_state.dtype))


class RandomVerticalFlip(Operation):
    """Flip the image vertically with probability flip_prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    flip_prob : float
        The probability with which to flip each image in the batch
        horizontally.
    """

    def __init__(self, flip_prob: float = 0.5):
        super().__init__()
        self.flip_prob = flip_prob

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        flip_prob = self.flip_prob

        def flip(images, dst):
            should_flip = rand(images.shape[0]) < flip_prob
            for i in my_range(images.shape[0]):
                if should_flip[i]:
                    dst[i] = images[i, ::-1]
                else:
                    dst[i] = images[i]

            return dst

        flip.is_parallel = True
        return flip

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True),
                AllocationQuery(previous_state.shape, previous_state.dtype))