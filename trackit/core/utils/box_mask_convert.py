import numpy as np


def rect_from_mask(mask: np.ndarray):
    '''
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    '''
    assert mask.ndim == 2
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return np.array((x0, y0, x1 + 1, y1 + 1), dtype=int)


def mask_from_rect(rect, output_size):
    '''
    create a binary mask from a given rectangle
    rect: axis-aligned rectangle [x0, y0, width, height]
    output_sz: size of the output [width, height]
    '''
    mask = np.zeros((output_size[1], output_size[0]), dtype=bool)
    x0, y0, x1, y1 = rect
    mask[y0:y1, x0:x1] = 1
    return mask
