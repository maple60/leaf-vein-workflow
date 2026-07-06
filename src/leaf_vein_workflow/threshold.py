"""Image thresholding helpers independent of napari."""

from typing import SupportsFloat

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a 2D grayscale float array from a supported image array."""
    image_array = np.asarray(image)

    if image_array.ndim == 2:
        grayscale = image_array.astype(float, copy=False)
    elif image_array.ndim == 3 and image_array.shape[-1] in (3, 4):
        rgb = image_array[..., :3].astype(float, copy=False)
        grayscale = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )
    else:
        raise ValueError("Only 2D grayscale and 2D RGB/RGBA images are supported.")

    if grayscale.size == 0:
        raise ValueError("Image is empty.")

    if not np.all(np.isfinite(grayscale)):
        raise ValueError("Image contains non-finite values.")

    return grayscale


def otsu_threshold(image: np.ndarray, bins: int = 256) -> float:
    """Estimate an Otsu threshold from a supported image array."""
    grayscale = to_grayscale(image)
    values = grayscale.ravel()

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        return minimum

    counts, bin_edges = np.histogram(values, bins=bins, range=(minimum, maximum))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    cumulative_counts = np.cumsum(counts)
    cumulative_sum = np.cumsum(counts * centers)
    total_count = cumulative_counts[-1]
    total_sum = cumulative_sum[-1]

    background_count = cumulative_counts
    foreground_count = total_count - cumulative_counts
    valid = (background_count > 0) & (foreground_count > 0)
    if not np.any(valid):
        return minimum

    background_mean = np.zeros_like(centers, dtype=float)
    foreground_mean = np.zeros_like(centers, dtype=float)
    background_mean[valid] = cumulative_sum[valid] / background_count[valid]
    foreground_mean[valid] = (
        total_sum - cumulative_sum[valid]
    ) / foreground_count[valid]

    between_class_variance = np.zeros_like(centers, dtype=float)
    between_class_variance[valid] = (
        background_count[valid]
        * foreground_count[valid]
        * (background_mean[valid] - foreground_mean[valid]) ** 2
    )

    return float(centers[int(np.argmax(between_class_variance))])


def threshold_image(
    image: np.ndarray,
    threshold: SupportsFloat,
    *,
    invert: bool = False,
) -> np.ndarray:
    """Return a boolean foreground mask from a supported image array."""
    grayscale = to_grayscale(image)
    threshold_value = float(threshold)

    if invert:
        return grayscale <= threshold_value
    return grayscale >= threshold_value
