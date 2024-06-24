from typing import Tuple, Union

import kornia.augmentation as K
from kornia.constants import Resample
from torch import Tensor
from torch import nn


class MAETransform(nn.Sequential):
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
        self, input_size: int = 112, min_scale: float = 0.2
    ):
        super().__init__(
            K.RandomResizedCrop(
                (input_size, input_size), scale=(min_scale, 1.0), resample=Resample.BICUBIC
            ),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(), # addition that is not in paper
        )
        self.input_size = input_size

    def forward(self, input: Tensor) -> list[Tensor]:
        return [super().forward(input)]