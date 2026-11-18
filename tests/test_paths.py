import csv

import numpy as np
import pytest

from leaf_vein_workflow.paths import (
    export_vein_paths,
    measure_vein_paths,
    path_length_px,
)


def test_path_length_sums_segments():
    path = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]])

    assert path_length_px(path) == pytest.approx(11.0)
    assert path_length_px(np.array([[1.0, 1.0]])) == 0.0


def test_measure_vein_paths_sums_orders_and_computes_vla():
    paths = [
        np.array([[0.0, 0.0], [100.0, 0.0]]),
        np.array([[10.0, 0.0], [10.0, 30.0]]),
        np.array([[20.0, 0.0], [20.0, 20.0]]),
    ]

    rows = measure_vein_paths(
        paths,
        [1, 2, 2],
        pixel_size_mm=0.1,
        leaf_area_mm2=50.0,
        image_name="leaf",
    )
    by_name = {row["label_name"]: row for row in rows}

    assert by_name["primary"]["n_paths"] == 1
    assert by_name["primary"]["length_mm"] == pytest.approx(10.0)
    assert by_name["primary"]["vla_mm_per_mm2"] == pytest.approx(0.2)
    assert by_name["secondary"]["n_paths"] == 2
    assert by_name["secondary"]["length_mm"] == pytest.approx(5.0)
    assert by_name["primary_secondary"]["length_mm"] == pytest.approx(15.0)
    assert by_name["primary_secondary"]["vla_mm_per_mm2"] == pytest.approx(0.3)
    assert all(row["method"] == "path" for row in rows)


def test_measure_vein_paths_without_calibration_or_area_leaves_vla_empty():
    rows = measure_vein_paths([np.array([[0.0, 0.0], [0.0, 5.0]])], [1])

    assert rows[0]["length_px"] == pytest.approx(5.0)
    assert rows[0]["length_mm"] is None
    assert rows[0]["vla_mm_per_mm2"] is None


def test_measure_vein_paths_rejects_unsupported_orders():
    with pytest.raises(ValueError, match="orders 1 and 2"):
        measure_vein_paths([np.zeros((2, 2))], [3])


def test_export_vein_paths_writes_measurements_and_vertices(tmp_path):
    paths = [np.array([[0.0, 0.0], [0.0, 10.0], [5.0, 10.0]])]
    rows = measure_vein_paths(paths, [1], pixel_size_mm=0.1)

    files = export_vein_paths(tmp_path, rows, paths, [1], prefix="leaf")

    with files.vertices_csv.open(encoding="utf-8") as file:
        vertices = list(csv.DictReader(file))
    assert len(vertices) == 3
    assert vertices[2] == {
        "path_id": "0",
        "order": "1",
        "vertex": "2",
        "y_px": "5.0",
        "x_px": "10.0",
    }
    assert "primary_secondary" in files.measurements_csv.read_text(encoding="utf-8")
