from typing import Optional, Tuple, Union

import kornia.augmentation as K
from torch import nn

from methods.transforms.base import MultiViewTransform


class BarlowTwinsView1Transform(nn.Sequential):
    def __init__(
        self,
        input_size: int = 112,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.2,
        cj_hue: float = 0.1,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 1.0,
        solarization_prob: float = 0.0,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
    ):

        super().__init__(
            K.RandomResizedCrop(size=(input_size, input_size), scale=(min_scale, 1.0)),
            K.RandomRotation(p=rr_prob, degrees=rr_degrees),
            K.RandomHorizontalFlip(p=hf_prob),
            K.RandomVerticalFlip(p=vf_prob),
            K.ColorJitter(
                brightness=cj_strength * cj_bright,
                contrast=cj_strength * cj_contrast,
                saturation=cj_strength * cj_sat,
                hue=cj_strength * cj_hue,
                p=cj_prob,
            ),
            # K.RandomGrayscale(p=random_gray_scale), # -> not useful for Earth Observation?
            K.RandomGaussianBlur(
                kernel_size=input_size // 10,
                sigma=sigmas,
                p=gaussian_blur,
                border_type="same",
            ),
            K.RandomSolarize(p=solarization_prob),
        )
        self.input_size = input_size


class BarlowTwinsView2Transform(nn.Sequential):
    def __init__(
        self,
        input_size: int = 112,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.2,
        cj_hue: float = 0.1,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.1,
        solarization_prob: float = 0.2,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
    ):
        super().__init__(
            K.RandomResizedCrop(size=(input_size, input_size), scale=(min_scale, 1.0)),
            K.RandomRotation(p=rr_prob, degrees=rr_degrees),
            K.RandomHorizontalFlip(p=hf_prob),
            K.RandomVerticalFlip(p=vf_prob),
            K.ColorJitter(
                brightness=cj_strength * cj_bright,
                contrast=cj_strength * cj_contrast,
                saturation=cj_strength * cj_sat,
                hue=cj_strength * cj_hue,
                p=cj_prob,
            ),
            # K.RandomGrayscale(p=random_gray_scale), # -> not useful for Earth Observation?
            K.RandomGaussianBlur(
                kernel_size=input_size // 10,
                sigma=sigmas,
                p=gaussian_blur,
                border_type="same",
            ),
            K.RandomSolarize(p=solarization_prob),
        )
        self.input_size = input_size


class BarlowTwinsTransform(MultiViewTransform):
    """Implements the transformations for BYOL[0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - Solarization

    Note that SimCLR v1 and v2 use similar augmentations. In detail, BYOL has
    asymmetric gaussian blur and solarization. Furthermore, BYOL has weaker
    color jitter compared to SimCLR.

    - [0]: Bootstrap Your Own Latent, 2020, https://arxiv.org/pdf/2006.07733.pdf

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of [tensor, tensor].

    Attributes:
        view_1_transform: The transform for the first view.
        view_2_transform: The transform for the second view.
        input_size:
    """

    def __init__(
        self,
        view_1_transform: BarlowTwinsView1Transform,
        view_2_transform: BarlowTwinsView2Transform,
    ):
        # We need to initialize the transforms here
        view_1_transform = view_1_transform or BarlowTwinsView1Transform()
        view_2_transform = view_2_transform or BarlowTwinsView2Transform()
        super().__init__(view_transforms=[view_1_transform, view_2_transform])
