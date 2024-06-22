from dataclasses import replace
from typing import List, Tuple, Union, Callable, Optional

import torchvision.transforms as T
from PIL.Image import Image
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from torch import Tensor


class MAETransform(Operation):
    """Implements the view augmentation for MAE [0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 1.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377

    Attributes:
        input_size:
            Size of the input image in pixels.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.

    """

    def __init__(
        self, input_size: Union[int, Tuple[int, int]] = 224, min_scale: float = 0.2
    ):
        super().__init__()
        transforms = [
            T.RandomResizedCrop(
                input_size, scale=(min_scale, 1.0), interpolation=3
            ),  # 3 is bicubic
            T.RandomHorizontalFlip(),
        ]

        self.input_size = input_size
        self.transform = T.Compose(transforms)

    def __call__(self, image: Union[Tensor, Image]):
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return [transformed]

    def generate_code(self) -> Callable:
        def transform(image: Union[Tensor, Image], _) -> List[Tensor]:
            """
            Applies the transforms to the input image.

            Args:
                image:
                    The input image to apply the transforms to.

            Returns:
                The transformed image.

            """
            return self.__call__(image)

        return transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (
            replace(
                previous_state,
                shape=(previous_state.shape[0], self.input_size, self.input_size)
            ),
            None,
        )
