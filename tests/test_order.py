import numpy as np
import pandas as pd
import pytest

from leaf_vein_workflow.order import (
    create_empty_order_labels,
    export_order_results,
    make_qc_overlay,
    measure_order_labels,
    validate_order_labels,
)


def test_create_empty_order_labels_matches_mask_shape():
    labels = create_empty_order_labels((3, 4))

    assert labels.dtype == np.uint8
    assert labels.shape == (3, 4)
    assert not np.any(labels)


def test_measure_order_labels_reports_order_lengths_and_vla():
    labels = np.zeros((7, 7), dtype=np.uint8)
    labels[1, 1:4] = 1
    labels[3, 1:5] = 2
    labels[5, 1:4] = 3
    leaf_mask = np.ones_like(labels, dtype=np.uint8)

    rows = measure_order_labels(
        labels,
        leaf_mask=leaf_mask,
        pixel_size_mm=0.01,
        image_name="leaf",
    )
    by_name = {row["label_name"]: row for row in rows}

    assert by_name["primary"]["area_px"] == 3
    assert by_name["secondary"]["area_px"] == 4
    assert by_name["minor"]["area_px"] == 3
    assert by_name["uncertain"]["length_px"] == 0.0
    assert by_name["major"]["area_px"] == 7
    assert by_name["primary"]["length_px"] > 0
    assert by_name["secondary"]["length_px"] > 0
    assert by_name["minor"]["length_px"] > 0
    assert by_name["primary"]["leaf_area_px"] == labels.size
    assert by_name["primary"]["vla_px_per_px2"] is not None
    assert by_name["primary"]["length_mm"] is not None
    assert by_name["primary"]["vla_mm_per_mm2"] is not None


def test_measure_order_labels_handles_empty_labels():
    labels = np.zeros((4, 4), dtype=np.uint8)

    rows = measure_order_labels(labels)

    assert {row["label_name"] for row in rows} == {
        "primary",
        "secondary",
        "minor",
        "uncertain",
        "major",
    }
    assert all(row["area_px"] == 0 for row in rows)
    assert all(row["skeleton_pixels"] == 0 for row in rows)
    assert all(row["length_px"] == 0.0 for row in rows)
    assert all(row["leaf_area_px"] is None for row in rows)


def test_measure_order_labels_rejects_non_finite_pixel_size():
    labels = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        measure_order_labels(labels, pixel_size_mm=np.nan)


def test_validate_order_labels_detects_mask_outside_unlabeled_and_invalid_values():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1:3] = 1
    labels = np.zeros_like(mask)
    labels[1, 1] = 1
    labels[0, 0] = 2
    labels[3, 3] = 9

    result = validate_order_labels(labels, mask)

    assert result.outside_mask_pixels == 2
    assert result.unlabeled_vein_pixels == 1
    assert result.label_pixel_counts[1] == 1
    assert result.label_pixel_counts[2] == 1
    assert result.label_pixel_counts[9] == 1
    assert result.invalid_label_values == [9]


def test_make_qc_overlay_returns_rgb_uint8_image():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    skeleton = np.zeros_like(mask)
    skeleton[2, 1:3] = 1
    labels = np.zeros_like(mask)
    labels[1, 1] = 1
    labels[1, 2] = 2

    overlay = make_qc_overlay(None, mask, skeleton, labels)

    assert overlay.dtype == np.uint8
    assert overlay.shape == (4, 4, 3)
    assert np.any(overlay > 0)


def test_export_order_results_writes_expected_files(tmp_path):
    labels = np.zeros((4, 4), dtype=np.uint8)
    overlay = np.zeros((4, 4, 3), dtype=np.uint8)
    measurements = measure_order_labels(labels, image_name="leaf")

    paths = export_order_results(
        tmp_path,
        measurements,
        labels,
        branch_table=pd.DataFrame({"branch_id": [0]}),
        qc_overlay=overlay,
        prefix="leaf",
    )

    assert paths.measurements_csv.exists()
    assert paths.order_labels_npy.exists()
    assert paths.qc_overlay_png.exists()
    assert paths.branch_table_csv.exists()
    assert "label_name" in paths.measurements_csv.read_text(encoding="utf-8")
    np.testing.assert_array_equal(np.load(paths.order_labels_npy), labels)
