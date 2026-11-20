"""Image thresholding helpers independent of napari."""

from dataclasses import dataclass
from typing import SupportsFloat

import numpy as np
from scipy import ndimage
from skimage.filters import gaussian
from skimage.morphology import (
    closing,
    disk,
    opening,
    remove_small_holes,
    remove_small_objects,
)


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
    return _otsu_values(to_grayscale(image).ravel(), bins=bins)


@dataclass(slots=True)
class VeinThreshold:
    """Otsu result that says which side is vein and where the leaf is."""

    threshold: float
    invert: bool
    vein_fraction: float
    leaf_threshold: float | None = None
    leaf_mask: np.ndarray | None = None

    @property
    def vein_side(self) -> str:
        return "dark" if self.invert else "bright"


def estimate_vein_threshold(
    image: np.ndarray,
    *,
    exclude_background: bool = True,
    leaf_mask: np.ndarray | None = None,
) -> VeinThreshold:
    """Estimate a vein threshold and decide which side of it is vein.

    With ``exclude_background``, a first Otsu split separates the leaf from the
    background (the class covering most of the image border).  A second Otsu
    split inside the leaf then separates veins from the lamina.  Veins are
    taken to be the smaller of the two classes, so ``invert`` is set when the
    darker pixels are the minority.
    """
    grayscale = to_grayscale(image)
    leaf_threshold = None
    values = grayscale.ravel()

    if leaf_mask is not None:
        leaf_mask = np.asarray(leaf_mask, dtype=bool)
        if leaf_mask.shape != grayscale.shape or not np.any(leaf_mask):
            raise ValueError("Leaf mask must match the image and contain leaf pixels.")
    elif exclude_background:
        leaf_threshold, leaf_mask = estimate_leaf_mask(image)
    if leaf_mask is not None:
        values = grayscale[leaf_mask]

    threshold = _otsu_values(values)
    dark_fraction = float(np.mean(values <= threshold))
    invert = dark_fraction <= 0.5
    vein_fraction = dark_fraction if invert else 1.0 - dark_fraction
    return VeinThreshold(
        threshold=threshold,
        invert=invert,
        vein_fraction=vein_fraction,
        leaf_threshold=leaf_threshold,
        leaf_mask=leaf_mask,
    )


def estimate_leaf_mask(image: np.ndarray) -> tuple[float, np.ndarray | None]:
    """Return a first Otsu split and a leaf candidate, if background is found.

    This is a candidate for manual review, not a confirmed leaf-area mask.
    """
    grayscale = to_grayscale(image)
    threshold = _otsu_values(grayscale.ravel())
    bright = grayscale > threshold
    background = bright if _border(bright).mean() >= 0.5 else ~bright
    return threshold, ~background if _looks_like_background(background) else None


def _border(array: np.ndarray) -> np.ndarray:
    return np.concatenate((array[0, :], array[-1, :], array[:, 0], array[:, -1]))


def _looks_like_background(background: np.ndarray) -> bool:
    """Return whether a class is paper around the leaf, not lamina between veins.

    Paper around a leaf is mostly connected to the ROI edge.  Inside a leaf
    blade, the lighter class is lamina split by veins into many enclosed
    areoles, most of which do not reach the edge.
    """
    fraction = float(background.mean())
    if not 0.05 <= fraction <= 0.95:
        return False
    labels, _ = ndimage.label(background)
    edge_labels = np.unique(_border(labels))
    edge_labels = edge_labels[edge_labels > 0]
    touching = np.isin(labels, edge_labels)
    return float(touching.sum()) / float(background.sum()) >= 0.8


def _otsu_values(values: np.ndarray, bins: int = 256) -> float:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("Image is empty.")

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


def smooth_image(image: np.ndarray, sigma: SupportsFloat) -> np.ndarray:
    """Return a Gaussian-smoothed 2D grayscale image."""
    grayscale = to_grayscale(image)
    sigma_value = float(sigma)

    if sigma_value < 0:
        raise ValueError("Gaussian sigma must be non-negative.")

    if sigma_value == 0:
        return grayscale.copy()

    return gaussian(
        grayscale,
        sigma=sigma_value,
        preserve_range=True,
        channel_axis=None,
    ).astype(float, copy=False)


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


def postprocess_mask(
    mask: np.ndarray,
    *,
    small_object_size: SupportsFloat = 100,
    max_hole_size: SupportsFloat = 500,
    closing_radius: SupportsFloat = 1,
    opening_radius: SupportsFloat = 0,
) -> np.ndarray:
    """Return a cleaned 0/1 mask after explicit morphology settings."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError("Only 2D masks can be postprocessed.")

    if mask_array.size == 0:
        raise ValueError("Mask is empty.")

    small_object_size_value = _nonnegative_int(
        small_object_size,
        "Small object size",
    )
    max_hole_size_value = _nonnegative_int(max_hole_size, "Hole size")
    closing_radius_value = _nonnegative_int(closing_radius, "Closing radius")
    opening_radius_value = _nonnegative_int(opening_radius, "Opening radius")

    clean = mask_array > 0

    if small_object_size_value > 0:
        clean = remove_small_objects(clean, max_size=small_object_size_value)

    if max_hole_size_value > 0:
        clean = remove_small_holes(clean, max_size=max_hole_size_value)

    if closing_radius_value > 0:
        clean = closing(clean, footprint=disk(closing_radius_value))

    if opening_radius_value > 0:
        clean = opening(clean, footprint=disk(opening_radius_value))

    return clean.astype(np.uint8)


def _nonnegative_int(value: SupportsFloat, name: str) -> int:
    int_value = int(value)
    if int_value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return int_value
