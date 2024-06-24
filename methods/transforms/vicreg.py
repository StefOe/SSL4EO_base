from typing import Optional, Tuple, Union

import kornia.augmentation as K
from torch import nn

from methods.transforms.base import MultiViewTransform


class VICRegTransform(MultiViewTransform):
    """Implements the transformations for VICReg.

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Random solarization
        - Gaussian blur
        - ImageNet normalization

    Similar to SimCLR transform but with extra solarization.

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        solarize_prob:
            Probability of solarization.
        gaussian_blur:
            Probability of Gaussian blur.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.

    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        solarize_prob: float = 0.1,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
    ):
        view_transform = VICRegViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            solarize_prob=solarize_prob,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
        )
        super().__init__(view_transforms=[view_transform, view_transform])
        self.input_size = input_size


class VICRegViewTransform(nn.Sequential):
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        solarize_prob: float = 0.1,
        gaussian_blur: float = 0.5,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
    ):
        rr_degrees = 90. if rr_degrees is None else rr_degrees

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
            K.RandomSolarize(p=solarize_prob),
            K.RandomGaussianBlur(
                kernel_size=input_size // 10,
                sigma=sigmas,
                p=gaussian_blur,
                border_type="same",
            ),
        )
