"""Source-image scale calibration for the napari workflow."""

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_info, show_warning

from ..calibration import (
    CALIBRATION_MM_PER_PIXEL_KEY,
    choose_scale_bar_length_mm,
    format_length_mm,
    scale_bar_segment,
    dpi_to_mm_per_pixel,
    line_length_pixels,
    make_calibration_metadata,
    micrometres_to_mm_per_pixel,
    millimetres_per_pixel,
    mm_per_pixel_to_dpi,
    scale_bar_to_mm_per_pixel,
)
from .controls import WorkflowControls, parse_number_input
from .layers import (
    ROLE_KEY,
    ROLE_SCALE_BAR,
    ROLE_SCALE_INDICATOR,
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
            self.controls.set_cleanup_unit_labels(None)
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
            self.controls.set_cleanup_unit_labels(None)
            return
        self._show_calibration(source_image, float(mm_per_pixel))

    def refresh_cleanup_unit_labels(self, *args) -> None:
        """Refresh physical cleanup equivalents after a pixel input changes."""
        source_image = self.layers.active_source_image()
        if source_image is None:
            return
        mm_per_pixel = source_image.metadata.get(CALIBRATION_MM_PER_PIXEL_KEY)
        if mm_per_pixel is None:
            self.controls.set_cleanup_unit_labels(None)
            return
        self.controls.set_cleanup_unit_labels(
            millimetres_per_pixel(float(mm_per_pixel))
        )

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
        input_unit: str
        pixel_length: float | None = None
        line_layer = None

        try:
            value = parse_number_input(
                self.controls.calibration_value_input,
                self.controls.calibration_value_label.text(),
            )
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
        self.show_scale_indicator(source_image, mm_per_pixel)
        self._show_calibration(source_image, mm_per_pixel)
        show_info(f"Applied scale calibration to {source_image.name}.")

    def show_scale_indicator(self, source_image: Image, mm_per_pixel: float) -> None:
        """Draw or update a round-length bar (10 mm when it fits) on the image."""
        shape = self._image_shape(source_image)
        if shape is None:
            return
        length_mm = choose_scale_bar_length_mm(shape[1], mm_per_pixel)
        segment = scale_bar_segment(shape, mm_per_pixel, length_mm)
        segment = segment + np.asarray(source_image.translate[-2:], dtype=float)
        label = format_length_mm(length_mm)
        edge_width = max(2.0, 0.006 * min(shape))

        indicator = self.layers.image_to_scale_indicator.get(id(source_image))
        if indicator is not None and self.layers.is_present(indicator):
            indicator.data = [segment]
            indicator.shape_type = ["line"]
            indicator.features = {"label": [label]}
            indicator.edge_width = [edge_width]
            desired_name = f"{source_image.name} {label} bar"
            if indicator.name != desired_name:
                indicator.name = self.layers.unique_name(desired_name)
            indicator.refresh()
            return

        previous_active = self.layers.viewer.layers.selection.active
        indicator = self.layers.viewer.add_shapes(
            [segment],
            shape_type="line",
            name=self.layers.unique_name(f"{source_image.name} {label} bar"),
            edge_color="magenta",
            edge_width=edge_width,
            features={"label": [label]},
            text={
                "string": "{label}",
                "color": "magenta",
                "size": 14,
                "anchor": "upper_left",
                "translation": [-3.0 * edge_width, 0.0],
            },
        )
        indicator.metadata[ROLE_KEY] = ROLE_SCALE_INDICATOR
        indicator.metadata[SOURCE_IMAGE_KEY] = source_image.name
        indicator.editable = False
        self.layers.image_to_scale_indicator[id(source_image)] = indicator
        # 表示用レイヤーを追加しても、ユーザーの作業中レイヤーは変えない。
        if previous_active is not None and self.layers.is_present(previous_active):
            self.layers.activate(previous_active)

    @staticmethod
    def _image_shape(layer: Image) -> tuple[int, int] | None:
        data_shape = tuple(np.shape(layer.data))
        if getattr(layer, "rgb", False):
            data_shape = data_shape[:-1]
        if len(data_shape) < 2:
            return None
        return int(data_shape[-2]), int(data_shape[-1])

    def _show_calibration(self, source_image: Image, mm_per_pixel: float) -> None:
        micrometres = mm_per_pixel * 1000.0
        equivalent_dpi = mm_per_pixel_to_dpi(mm_per_pixel)
        self.controls.set_cleanup_unit_labels(mm_per_pixel)
        self.controls.set_calibration_status(
            f"Scale for {source_image.name}: 1 px = {micrometres:.6g} µm "
            f"({mm_per_pixel:.6g} mm); equivalent DPI ≈ {equivalent_dpi:.6g}"
        )

    def _warn(self, message: str) -> None:
        self.controls.set_calibration_status(message)
        show_warning(message)
