"""Widget integration tests: export."""

import numpy as np
import pytest

from widget_helpers import (
    _advance_to_mask,
)


def test_export_measurements_converts_leaf_area_units_and_uses_output_dir(
    widget_flow, tmp_path
):
    import csv
    from pathlib import Path

    flow = widget_flow
    assert Path(flow.widget.output_dir_input.text()) == Path.cwd() / "output"

    mm_index = flow.widget.calibration_method_combo.findData("mm_per_pixel")
    flow.widget.calibration_method_combo.setCurrentIndex(mm_index)
    flow.widget.calibration_value_input.setText("0.1")
    flow.widget.apply_scale()

    _, _, mask_layer = _advance_to_mask(flow)
    mask_layer.data = np.ones((2, 3), dtype=np.uint8)
    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    order_layer.data = np.array([[1, 1, 1], [3, 0, 4]], dtype=np.uint8)
    flow.app.processEvents()

    assert flow.widget.export_button.isEnabled()
    flow.widget.leaf_area_input.setText("2")
    rows = flow.widget.measure_veins()
    by_name = {row["label_name"]: row for row in rows}
    assert by_name["primary"]["length_mm"] == pytest.approx(0.2)
    assert by_name["primary"]["vla_mm_per_mm2"] == pytest.approx(0.1)
    assert by_name["secondary"]["length_mm"] == 0.0
    assert by_name["major"]["length_mm"] == pytest.approx(
        by_name["primary"]["length_mm"] + by_name["tertiary"]["length_mm"]
    )

    # 0.04 cm² = 4 mm²
    flow.widget.leaf_area_input.setText("0.04")
    flow.widget.leaf_area_unit_combo.setCurrentIndex(
        flow.widget.leaf_area_unit_combo.findData("cm2")
    )
    flow.widget.output_dir_input.setText(str(tmp_path / "out"))
    status_before = flow.widget.status_label.text()

    result = flow.widget.export_measurements()
    assert result.paths is None
    paths = result.labels

    assert paths.measurements_csv.parent.parent == tmp_path / "out"
    assert paths.measurements_csv.parent.name in paths.measurements_csv.name
    with paths.measurements_csv.open(encoding="utf-8") as file:
        exported = {row["label_name"]: row for row in csv.DictReader(file)}
    assert float(exported["primary"]["leaf_area_mm2"]) == pytest.approx(4.0)
    assert float(exported["primary"]["vla_mm_per_mm2"]) == pytest.approx(0.05)
    assert "tertiary" in exported
    assert paths.qc_overlay_png.exists()
    assert paths.branch_table_csv.exists()
    assert flow.widget.status_label.text() == status_before


def test_export_qc_shows_only_measured_primary_path(tmp_path):
    from skimage.io import imread
    from leaf_vein_workflow.widgets.measurement import (
        ExportJob, LabelsSnapshot, run_export_job,
    )

    labels = np.zeros((85, 85), dtype=np.uint8)
    labels[8:73, 15:22] = 1
    labels[35, 22:75] = 1
    snapshot = LabelsSnapshot(
        labels=labels, vein_mask=labels > 0, base_image=None,
        mm_per_pixel=0.1, image_name="leaf", layer_name="orders",
        primary_endpoints_rc=((8, 18), (72, 18)),
    )
    job = ExportJob(
        target=tmp_path, prefix="leaf", leaf_area_mm2=10.0,
        labels=snapshot, paths=None,
        metadata={"image_name": "leaf", "leaf_area": {"source": "input"}},
        source_path=None,
    )
    result = run_export_job(job)
    overlay = imread(result.labels.qc_overlay_png)
    white = np.all(overlay == 255, axis=-1)
    assert white[:, 15:22].any()
    assert not white[:, 23:].any()
    np.testing.assert_array_equal(np.load(result.labels.order_labels_npy), labels)


def test_measure_veins_warns_without_order_labels(widget_flow):
    flow = widget_flow
    _advance_to_mask(flow)

    assert flow.widget.measure_veins() is None
    assert "order labels" in flow.widget.status_label.text()


def test_export_counts_mask_pixels_without_order_label(widget_flow, tmp_path):
    flow = widget_flow
    _, _, mask_layer = _advance_to_mask(flow)
    mask_layer.data = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    # 次数ラベルを消しゴムで0にした（maskには残っている）状態。
    order_layer.data = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    flow.app.processEvents()

    flow.widget.export_measurements(tmp_path)

    assert flow.widget._measurement_workflow.unlabeled_vein_pixels == 1


def test_vein_paths_take_selected_order_and_export_lengths(widget_flow, tmp_path):
    import csv

    flow = widget_flow
    image = flow.viewer.add_image(np.zeros((200, 300)), name="whole")
    flow.app.processEvents()
    flow.viewer.layers.selection.active = image
    index = flow.widget.calibration_method_combo.findData("mm_per_pixel")
    flow.widget.calibration_method_combo.setCurrentIndex(index)
    flow.widget.calibration_value_input.setText("0.1")
    flow.widget.apply_scale()
    flow.viewer.layers.selection.active = image

    paths_layer = flow.widget.draw_vein_paths()
    assert paths_layer.name == "whole vein paths"
    assert paths_layer.mode == "add_path"
    assert paths_layer.metadata[flow.module.CALIBRATION_MM_PER_PIXEL_KEY] == 0.1

    paths_layer.add_paths([np.array([[10.0, 10.0], [110.0, 10.0]])])
    flow.widget.path_order_combo.setCurrentIndex(
        flow.widget.path_order_combo.findData(2)
    )
    paths_layer.add_paths([np.array([[20.0, 10.0], [20.0, 40.0]])])
    paths_layer.add_paths([np.array([[30.0, 10.0], [30.0, 30.0]])])
    assert paths_layer.features["order"].tolist() == [1, 2, 2]

    # 選択したパスの次数を付け替える。
    paths_layer.selected_data = {2}
    flow.widget.path_order_combo.setCurrentIndex(
        flow.widget.path_order_combo.findData(1)
    )
    assert paths_layer.features["order"].tolist() == [1, 2, 1]

    flow.widget.leaf_area_input.setText("1")
    flow.widget.leaf_area_unit_combo.setCurrentIndex(
        flow.widget.leaf_area_unit_combo.findData("cm2")
    )
    result = flow.widget.export_measurements(tmp_path)

    assert result.labels is None
    with result.paths.measurements_csv.open(encoding="utf-8") as file:
        rows = {row["label_name"]: row for row in csv.DictReader(file)}
    assert float(rows["primary"]["length_mm"]) == pytest.approx(12.0)
    assert float(rows["secondary"]["length_mm"]) == pytest.approx(3.0)
    assert float(rows["primary_secondary"]["vla_mm_per_mm2"]) == pytest.approx(0.15)
    assert result.paths.vertices_csv.exists()


def test_export_metadata_records_actual_processing(widget_flow, tmp_path):
    import yaml

    flow = widget_flow
    dpi_index = flow.widget.calibration_method_combo.findData("dpi")
    flow.widget.calibration_method_combo.setCurrentIndex(dpi_index)
    flow.widget.calibration_value_input.setText("254")
    flow.widget.apply_scale()

    flow.widget.gaussian_sigma_spinbox.setValue(0.5)
    _, _, mask_layer = _advance_to_mask(flow)
    flow.widget.max_hole_size_spinbox.setValue(3)
    flow.widget.clean_mask()
    # Clean Mask の後に手で1画素足す。
    edited = np.asarray(mask_layer.data).copy()
    edited[0, 0] = 1
    edited[0, 1] = 1
    mask_layer.data = edited
    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    labels = np.asarray(order_layer.data).copy()
    labels[labels == 4] = 1
    order_layer.data = labels
    flow.app.processEvents()
    flow.widget.leaf_area_input.setText("2.5")
    flow.widget.leaf_area_unit_combo.setCurrentIndex(
        flow.widget.leaf_area_unit_combo.findData("cm2")
    )

    result = flow.widget.export_measurements(tmp_path)

    meta = yaml.safe_load(result.metadata_yaml.read_text(encoding="utf-8"))
    assert meta["image_name"] == "leaf"
    assert meta["calibration"]["method"] == "dpi"
    assert meta["calibration"]["input_value"] == 254.0
    assert meta["leaf_area"] == {"input": "2.5", "unit": "cm2", "mm2": 250.0}
    processing = meta["order_labels"]["processing"]
    assert processing["roi"] == {
        "row_start": 1, "row_stop": 3, "col_start": 1, "col_stop": 4,
    }
    assert processing["gaussian_sigma"] == 0.5
    assert "otsu_threshold" in processing
    assert set(processing["threshold"]) == {"value", "invert", "inside_leaf_only"}
    assert len(processing["cleanup"]) == 1
    assert processing["cleanup"][0]["max_hole_px"] == 3
    assert processing["mask_edited_by_hand_after_last_step"] is True
    assert meta["order_labels"]["uncertain_px"] == 0
    assert "defaults_in_effect" in meta["settings"]
    assert "napari" in meta["software"]

    prefix = result.metadata_yaml.name.removesuffix("_metadata.yaml")
    assert result.vein_mask_npy.name == f"{prefix}_vein_mask.npy"
    np.testing.assert_array_equal(np.load(result.vein_mask_npy), edited)
    for name in meta["files"]:
        assert name.startswith(prefix)
        assert (result.metadata_yaml.parent / name).exists()


def test_export_button_runs_in_background(widget_flow, tmp_path):
    from qtpy.QtCore import QEventLoop, QTimer

    flow = widget_flow
    flow.widget.calibration_value_input.setText("300")
    flow.widget.apply_scale()
    _advance_to_mask(flow)
    flow.widget.create_order_labels()
    flow.widget.leaf_area_input.setText("10")
    flow.widget._measurement_workflow.set_output_dir(tmp_path)

    worker = flow.widget.export_measurements(background=True)
    assert worker is not None
    assert not flow.widget.export_button.isEnabled()

    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(20000, loop.quit)
    if flow.widget._export_worker is not None:
        loop.exec_()
    flow.app.processEvents()

    assert flow.widget.export_button.isEnabled()
    assert not flow.widget._controls.progress_bar.isVisibleTo(flow.widget)
    assert list(tmp_path.glob("*/*_measurements.csv"))
    assert list(tmp_path.glob("*/*_metadata.yaml"))


def test_export_folders_do_not_overwrite_previous_run(widget_flow, tmp_path, monkeypatch):
    import leaf_vein_workflow.widgets.measurement as measurement
    flow = widget_flow
    _, _, mask = _advance_to_mask(flow)
    mask.data = np.ones((2, 3), dtype=np.uint8)
    flow.widget.create_order_labels()
    flow.widget.leaf_area_input.setText("10")
    monkeypatch.setattr(measurement, "_safe_prefix", lambda name: "fixed_leaf")
    first = flow.widget.export_measurements(tmp_path)
    original = first.metadata_yaml.read_bytes()
    second = flow.widget.export_measurements(tmp_path)
    assert first.metadata_yaml.parent == tmp_path / "fixed_leaf"
    assert second.metadata_yaml.parent == tmp_path / "fixed_leaf_2"
    assert first.metadata_yaml.read_bytes() == original
