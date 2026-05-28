import numpy as np


def expand_bounding_box_to_match_multiple_of(bounding_box: np.ndarray, multiple: int, square: bool = False) -> np.ndarray:
    '''
    Expand a bounding box to match the multiple of a number.

    Parameters:
        bounding_box: numpy array, the input bounding box (shape: (4,)), in the format (x1, y1, x2, y2)
        multiple: int, the number to match the multiple of

    Returns:
        numpy array, the expanded bounding box
    '''
    assert bounding_box.ndim == 1
    assert bounding_box.size == 4
    assert isinstance(multiple, int)
    assert multiple > 0

    x1, y1, x2, y2 = bounding_box.tolist()
    x1 = (x1 // multiple) * multiple
    y1 = (y1 // multiple) * multiple
    eps = 1.e-3
    x2 = ((x2 + multiple - eps) // multiple) * multiple
    y2 = ((y2 + multiple - eps) // multiple) * multiple
    if square:
        max_dimension = max(x2 - x1, y2 - y1)
        x2 = x1 + max_dimension
        y2 = y1 + max_dimension
    return np.array([x1, y1, x2, y2], dtype=bounding_box.dtype)
