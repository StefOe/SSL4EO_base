from typing import List, Tuple, Union

import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor
from torch.nn import Module

from methods.transforms.base import to_tensor


class MAETransform(Module):
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
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        min_scale: float = 0.2,
    ):
        super().__init__()
        transforms = [
            to_tensor,
            T.RandomResizedCrop(
                input_size, scale=(min_scale, 1.0), interpolation=3
            ),  # 3 is bicubic
            T.RandomHorizontalFlip(),
        ]

        self.transform = T.Compose(transforms)

    def forward(self, image: Union[Tensor, Image]) -> List[Tensor]:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        return [self.transform(image)]
