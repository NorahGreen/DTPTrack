from typing import Tuple

import numpy as np


def cosine_window(M: int, alpha: float = 0.5):
    fac = np.linspace(-np.pi, np.pi, M)
    return alpha + (1. - alpha) * np.cos(fac)


def cosine_window_object(original_size: int, crop_range: Tuple[float, float], object_center: float, M: int, alpha: float = 0.5):
    original_left = object_center / original_size * np.pi
    crop_box_left = crop_range[0] / original_size * np.pi + original_left
    crop_box_right = crop_range[1] / original_size * np.pi + original_left
    fac = np.linspace(crop_box_left, crop_box_right, M)
    return alpha + (1. - alpha) * np.cos(fac)

