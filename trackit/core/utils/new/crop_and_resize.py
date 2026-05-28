from dataclasses import dataclass
import torch
import numpy as np
from typing import Tuple, Optional

from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.core.operator.numpy.bbox.scale import bbox_scale


@dataclass(frozen=True)
class CropAndResizeResult:
    image: torch.Tensor
    foreground_bounding_box: np.ndarray
    scaling_ratio: Tuple[float, float]

    def get_foreground_mask(self, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if size is None:
            h, w = self.image.shape[-2:]
            size = w, h
        mask = torch.zeros((size[1], size[0]), dtype=torch.bool)
        image_h, image_w = self.image.shape[-2:]
        scaling_factor = np.array((size[0] / image_w, size[1] / image_h), dtype=float)
        foreground_bounding_box = bbox_scale(self.foreground_bounding_box, scaling_factor)
        foreground_bounding_box = bbox_rasterize(foreground_bounding_box)
        x1, y1, x2, y2 = foreground_bounding_box.tolist()
        mask[y1:y2, x1:x2] = True
        return mask


def crop_and_resize(image: torch.Tensor, bounding_box: np.ndarray, output_size: Tuple[int, int], background_fill_value=0.) -> CropAndResizeResult:
    assert image.ndim == 3  # Check image is in (C, H, W) format
    assert bounding_box.ndim == 1 and bounding_box.size == 4  # Check bounding box validity
    # assert np.issubdtype(bounding_box.dtype, np.integer)  # Check bounding box data type
    assert isinstance(output_size, (list, tuple)) and len(output_size) == 2 and all(dim > 0 for dim in output_size)  # Output size checks

    C, H, W = image.shape
    if not np.issubdtype(bounding_box.dtype, np.integer):
        from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
        bounding_box = bbox_rasterize(bounding_box).astype(int)
    x1, y1, x2, y2 = bounding_box.tolist()

    # Calculate dimensions of the crop that is within image bounds
    x1_clamped, y1_clamped = max(0, x1), max(0, y1)
    x2_clamped, y2_clamped = min(W, x2), min(H, y2)

    # Crop the image
    cropped_image = image[:, y1_clamped:y2_clamped, x1_clamped:x2_clamped]

    # Create the output image filled with the background value
    output_image = torch.full((C, y2 - y1, x2 - x1), background_fill_value, dtype=image.dtype, device=image.device)

    # Calculate where to place the cropped image in the output image
    target_y1 = max(0, -y1)  # Offset if y1 is out of bounds
    target_x1 = max(0, -x1)  # Offset if x1 is out of bounds
    target_y2 = target_y1 + (y2_clamped - y1_clamped)
    target_x2 = target_x1 + (x2_clamped - x1_clamped)

    # Place the cropped image in the output image
    output_image[:, target_y1:target_y2, target_x1:target_x2] = cropped_image

    # Resize the output image to desired output size
    resized_image = torch.nn.functional.interpolate(output_image.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False).squeeze(0)

    scaling_ratio = (output_size[0] / (x2 - x1), output_size[1] / (y2 - y1))
    foreground_bounding_box = target_x1 * scaling_ratio[0], target_y1 * scaling_ratio[1], target_x2 * scaling_ratio[0], target_y2 * scaling_ratio[1]
    foreground_bounding_box = np.array(foreground_bounding_box)

    return CropAndResizeResult(image=resized_image, foreground_bounding_box=foreground_bounding_box, scaling_ratio=scaling_ratio)
