"""Skeletonization helpers independent of napari."""

import numpy as np
from skimage.morphology import skeletonize


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Return a 0/1 skeleton image from a 2D binary or label mask."""
    mask_array = np.asarray(mask)

    if mask_array.ndim != 2:
        raise ValueError("Only 2D masks can be skeletonized.")

    if mask_array.size == 0:
        raise ValueError("Mask is empty.")

    foreground = mask_array > 0
    return skeletonize(foreground).astype(np.uint8)
