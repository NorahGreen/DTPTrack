import numpy as np
from typing import Tuple
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping_to_boxes
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_


def get_foreground_bounding_box(bbox: np.ndarray, siamfc_cropping_parameter: np.ndarray,
                                cropped_image_size: np.ndarray,
                                stride: Tuple[float, float]):
    siamfc_cropped_bbox = apply_siamfc_cropping_to_boxes(bbox, siamfc_cropping_parameter)
    bbox_clip_to_image_boundary_(siamfc_cropped_bbox, cropped_image_size)
    siamfc_cropped_bbox[0] /= stride[0]
    siamfc_cropped_bbox[1] /= stride[1]
    siamfc_cropped_bbox[2] /= stride[0]
    siamfc_cropped_bbox[3] /= stride[1]
    return bbox_rasterize(siamfc_cropped_bbox, dtype=np.int64)


def generate_soft_foreground_mask(bbox: np.ndarray, siamfc_cropping_parameter: np.ndarray, template_size: Tuple[int, int], stride: Tuple[float, float]):
    """
    Generate a soft foreground mask based on the bounding box and cropping parameters.

    :param bbox: Bounding box in the format [x1, y1, x2, y2].
    :param siamfc_cropping_parameter: Cropping parameters for SiamFC.
    :param template_feat_size: Size of the feature map (height, width).
    :param stride: Stride of the feature map (stride_height, stride_width).
    :return: Soft foreground mask as a 2D numpy array.
    """
    siamfc_cropped_bbox = apply_siamfc_cropping_to_boxes(bbox, siamfc_cropping_parameter)
    bbox_clip_to_image_boundary_(siamfc_cropped_bbox, np.asarray(template_size, dtype=np.float32))
    siamfc_cropped_bbox = bbox_rasterize(siamfc_cropped_bbox, dtype=np.int64)
    assert bbox_is_valid(siamfc_cropped_bbox)
    mask = np.zeros((template_size[1], template_size[0]), dtype=np.float32)
    mask[siamfc_cropped_bbox[1]: siamfc_cropped_bbox[3], siamfc_cropped_bbox[0]: siamfc_cropped_bbox[2]] = 1.0
    mask = mask.reshape((template_size[1] // stride[1], stride[1], template_size[0] // stride[0], stride[0])).mean(axis=(1, 3))
    return mask
