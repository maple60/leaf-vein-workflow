import numpy as np
import pytest
from skimage.io import imsave

from leaf_vein_workflow.provenance import array_digest, file_sha256, write_metadata
from leaf_vein_workflow.resume import MissingSourceImage, read_saved_analysis


def _saved_export(tmp_path):
    source = tmp_path / "scan.tif"
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    imsave(source, image, check_contrast=False)
    folder = tmp_path / "export"
    folder.mkdir()
    leaf = np.ones((6, 7), dtype=np.uint8)
    vein = leaf.copy()
    labels = np.full(leaf.shape, 4, dtype=np.uint8)
    labels[1, :] = 2
    labels[2, 2:5] = 3
    for suffix, values in (("leaf_mask", leaf), ("vein_mask", vein), ("vein_order_labels", labels)):
        np.save(folder / f"saved_{suffix}.npy", values)
    meta = write_metadata(folder / "saved_metadata.yaml", {
        "image_name": "scan", "source_image": {
            "path": str(source), "sha256": file_sha256(source), "shape": list(image.shape),
        },
        "calibration": {"mm_per_pixel": 0.1},
        "leaf_area": {"source": "leaf_mask", "sha256": array_digest(leaf), "mm2": 0.42},
        "order_labels": {"processing": {"gaussian_sigma": 1.0, "roi": {
            "row_start": 2, "row_stop": 8, "col_start": 1, "col_stop": 8,
        }}},
    })
    return folder, meta, source, labels


def test_reads_folder_and_legacy_metadata_path_without_changing_labels(tmp_path):
    folder, meta, _, labels = _saved_export(tmp_path)
    saved = read_saved_analysis(folder)
    np.testing.assert_array_equal(saved.labels, labels)
    assert saved.roi == (2, 8, 1, 8)
    np.testing.assert_array_equal(read_saved_analysis(meta).leaf, saved.leaf)


def test_relocated_original_can_be_selected_and_wrong_scan_is_rejected(tmp_path):
    folder, _, source, _ = _saved_export(tmp_path)
    moved = tmp_path / "moved.tif"
    source.rename(moved)
    with pytest.raises(MissingSourceImage):
        read_saved_analysis(folder)
    assert read_saved_analysis(folder, moved).source_path == moved
    wrong = tmp_path / "wrong.tif"
    imsave(wrong, np.zeros((10, 10), dtype=np.uint8), check_contrast=False)
    with pytest.raises(ValueError, match="checksum"):
        read_saved_analysis(folder, wrong)


@pytest.mark.parametrize("problem", ["missing", "shape", "values", "leaf_checksum", "outside_mask"])
def test_rejects_incomplete_or_inconsistent_masks(tmp_path, problem):
    folder, _, _, _ = _saved_export(tmp_path)
    file = folder / "saved_vein_order_labels.npy"
    if problem == "missing":
        file.unlink()
    elif problem == "shape":
        np.save(file, np.ones((2, 2), dtype=np.uint8))
    elif problem == "values":
        np.save(file, np.full((6, 7), 5, dtype=np.uint8))
    elif problem == "leaf_checksum":
        leaf = np.ones((6, 7), dtype=np.uint8)
        leaf[0, 0] = 0
        vein = leaf.copy()
        labels = np.load(file)
        labels[0, 0] = 0
        np.save(folder / "saved_leaf_mask.npy", leaf)
        np.save(folder / "saved_vein_mask.npy", vein)
        np.save(file, labels)
    else:
        np.save(folder / "saved_vein_mask.npy", np.zeros((6, 7), dtype=np.uint8))
    with pytest.raises(ValueError):
        read_saved_analysis(folder)
