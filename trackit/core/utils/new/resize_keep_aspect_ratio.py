from typing import Tuple, Optional
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class KeepAspectRatioResizingResult:
    image: torch.Tensor
    scaling_factor: float
    foreground_size_sub_pixel: Tuple[float, float]

    def get_foreground_mask(self, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        image_h, image_w = self.image.shape[-2:]
        if size is None:
            size = image_w, image_h
        mask = torch.zeros((size[1], size[0]), dtype=torch.bool)
        scaling_factor = (size[0] / image_w, size[1] / image_h)
        foreground_size = self.foreground_size_sub_pixel
        eps = 1.e-3
        foreground_size = (int(foreground_size[0] * scaling_factor[0] + 1 - eps), int(foreground_size[1] * scaling_factor[1] + 1 - eps))
        mask[:foreground_size[1], :foreground_size[0]] = True
        return mask


def resize_image_keep_aspect_ratio_get_scaling_factor(image_size: Tuple[int, int], output_size: Tuple[int, int]) -> float:
    '''
    Calculate the scaling factor to resize an image while keeping the aspect ratio.

    Parameters:
        image_size: tuple, the original image size (width, height)
        output_size: tuple, the target size (width, height)

    Returns:
        float, the scaling factor
    '''
    aspect_ratio = image_size[0] / image_size[1]
    if output_size[0] / output_size[1] > aspect_ratio:
        # Width is too large
        scaling_factor = output_size[1] / image_size[1]
    else:
        # Height is too large
        scaling_factor = output_size[0] / image_size[0]
    return scaling_factor


def resize_image_keep_aspect_ratio_torch(image: torch.Tensor, output_size: Tuple[int, int], background_fill_value=0.) -> KeepAspectRatioResizingResult:
    '''
    Resize an image while keeping the aspect ratio.

    Parameters:
        image: torch.Tensor, the input image (shape: (C, H, W))
        output_size: tuple, the target size (width, height)
        background_fill_value: float, the value to fill the background with

    Returns:
        torch.Tensor, the resized image (shape: (C, output_height, output_width))
    '''
    # Get original dimensions
    ori_height, ori_width = image.shape[-2:]

    # Calculate target size
    target_width, target_height = output_size
    aspect_ratio = ori_width / ori_height

    if target_width / target_height > aspect_ratio:
        # Width is too large
        scaling_factor = target_height / ori_height
        target_height_sub_pixel = target_height
        target_width_sub_pixel = target_height * aspect_ratio
    else:
        # Height is too large
        scaling_factor = target_width / ori_width
        target_width_sub_pixel = target_width
        target_height_sub_pixel = target_width / aspect_ratio

    # Resize the image
    resized_image = torch.nn.functional.interpolate(
        image.unsqueeze(0),  # Add batch dimension
        scale_factor=scaling_factor,
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Remove batch dimension

    target_height, target_width = resized_image.shape[-2:]

    if target_width != output_size[0] or target_height != output_size[1]:
        # Pad the image to the target size
        pad_width = output_size[0] - target_width
        pad_height = output_size[1] - target_height
        resized_image = torch.nn.functional.pad(
            resized_image.unsqueeze(0),  # Add batch dimension
            (0, pad_width, 0, pad_height),
            mode='constant',
            value=background_fill_value
        ).squeeze(0)

    return KeepAspectRatioResizingResult(resized_image, scaling_factor, (target_width_sub_pixel, target_height_sub_pixel))
