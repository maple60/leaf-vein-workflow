"""Source-image scale calibration for the napari workflow."""

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_info, show_warning

from ..calibration import (
    CALIBRATION_MM_PER_PIXEL_KEY,
    dpi_to_mm_per_pixel,
    line_length_pixels,
    make_calibration_metadata,
    micrometres_to_mm_per_pixel,
    millimetres_per_pixel,
    mm_per_pixel_to_dpi,
    scale_bar_to_mm_per_pixel,
)
from .controls import WorkflowControls
from .layers import (
    ROLE_KEY,
    ROLE_SCALE_BAR,
    SOURCE_IMAGE_KEY,
    WorkflowLayers,
)


class CalibrationWorkflow:
    """Calibrate each source image without changing napari display coordinates."""

    def __init__(self, layers: WorkflowLayers, controls: WorkflowControls):
        self.layers = layers
        self.controls = controls

    def prepare_image_layer(self, layer) -> None:
        """Enable calibration when a new unprocessed source image is inserted."""
        if not isinstance(layer, Image) or layer.metadata.get(ROLE_KEY) is not None:
            return
        self.controls.set_calibration_enabled(True)
        mm_per_pixel = layer.metadata.get(CALIBRATION_MM_PER_PIXEL_KEY)
        if mm_per_pixel is None:
            self.controls.set_calibration_status(
                f"Scale for {layer.name}: not calibrated (pixel units)"
            )
        else:
            self._show_calibration(layer, float(mm_per_pixel))

    def on_method_changed(self, *args) -> None:
        """Update labels and scale-bar controls for the selected method."""
        method = str(self.controls.calibration_method_combo.currentData())
        self.controls.set_calibration_method(method)

    def on_active_layer_changed(self, *args) -> None:
        """Keep the displayed scale synchronized in multi-image sessions."""
        source_image = self.layers.active_source_image()
        if source_image is None:
            return
        mm_per_pixel = source_image.metadata.get(CALIBRATION_MM_PER_PIXEL_KEY)
        if mm_per_pixel is None:
            self.controls.set_calibration_status(
                f"Scale for {source_image.name}: not calibrated (pixel units)"
            )
            return
        self._show_calibration(source_image, float(mm_per_pixel))

    def draw_scale_bar_line(self) -> None:
        """Create or activate one line layer linked to the selected source image."""
        source_image = self.layers.active_source_image()
        if source_image is None:
            self._warn("Select a source image or one of its workflow layers first.")
            return

        line_layer = self.layers.image_to_scale_bar.get(id(source_image))
        if line_layer is None or not self.layers.is_present(line_layer):
            line_layer = self.layers.viewer.add_shapes(
                name=self.layers.unique_name(f"{source_image.name} scale bar"),
                ndim=2,
                edge_color="cyan",
                face_color="transparent",
                edge_width=3,
            )
            line_layer.metadata.update(
                {
                    ROLE_KEY: ROLE_SCALE_BAR,
                    SOURCE_IMAGE_KEY: source_image.name,
                    **self.layers.inherited_calibration(source_image),
                }
            )
            self.layers.scale_bar_to_image[id(line_layer)] = source_image
            self.layers.image_to_scale_bar[id(source_image)] = line_layer

        # ROI確定後も、画像内のバーを見ながら再校正できるよう元画像を表示する。
        source_image.visible = True
        self.layers.activate(line_layer)
        line_layer.mode = "add_line"
        self.controls.set_calibration_status(
            f"Draw exactly one line along the scale bar in {source_image.name}."
        )
        show_info("Draw one line along the visible scale bar, then click Apply Scale.")

    def apply_scale(self) -> None:
        """Normalize the selected calibration method to millimetres per pixel."""
        source_image = self.layers.active_source_image()
        if source_image is None:
            self._warn("Select a source image or one of its workflow layers first.")
            return

        method = str(self.controls.calibration_method_combo.currentData())
        value = self.controls.calibration_value_spinbox.value()
        input_unit: str
        pixel_length: float | None = None
        line_layer = None

        try:
            if method == "dpi":
                mm_per_pixel = dpi_to_mm_per_pixel(value)
                input_unit = "dpi"
            elif method == "um_per_pixel":
                mm_per_pixel = micrometres_to_mm_per_pixel(value)
                input_unit = "um_per_pixel"
            elif method == "mm_per_pixel":
                mm_per_pixel = millimetres_per_pixel(value)
                input_unit = "mm_per_pixel"
            elif method == "scale_bar":
                line_layer = self.layers.image_to_scale_bar.get(id(source_image))
                if line_layer is None or not self.layers.is_present(line_layer):
                    raise ValueError(
                        "Draw a scale-bar line for the selected image first."
                    )
                if len(line_layer.data) != 1:
                    raise ValueError("Scale-bar layer must contain exactly one line.")
                if self.layers.single_shape_type(line_layer) != "line":
                    raise ValueError("Scale-bar shape must be a line.")
                pixel_length = line_length_pixels(np.asarray(line_layer.data[0]))
                input_unit = str(self.controls.calibration_unit_combo.currentData())
                mm_per_pixel = scale_bar_to_mm_per_pixel(
                    pixel_length,
                    value,
                    input_unit,
                )
            else:
                raise ValueError(f"Unsupported calibration method: {method}")

            metadata = make_calibration_metadata(
                mm_per_pixel=mm_per_pixel,
                method=method,
                input_value=value,
                input_unit=input_unit,
                scale_bar_pixel_length=pixel_length,
            )
        except ValueError as exc:
            self._warn(str(exc))
            return

        self.layers.propagate_calibration(source_image, metadata)
        if line_layer is not None:
            line_layer.mode = "select"
        self._show_calibration(source_image, mm_per_pixel)
        show_info(f"Applied scale calibration to {source_image.name}.")

    def _show_calibration(self, source_image: Image, mm_per_pixel: float) -> None:
        micrometres = mm_per_pixel * 1000.0
        equivalent_dpi = mm_per_pixel_to_dpi(mm_per_pixel)
        self.controls.set_calibration_status(
            f"Scale for {source_image.name}: 1 px = {micrometres:.6g} µm "
            f"({mm_per_pixel:.6g} mm); equivalent DPI ≈ {equivalent_dpi:.6g}"
        )

    def _warn(self, message: str) -> None:
        self.controls.set_calibration_status(message)
        show_warning(message)
