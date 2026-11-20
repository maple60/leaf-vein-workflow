import numpy as np
import yaml

from leaf_vein_workflow.provenance import (
    array_digest,
    file_sha256,
    software_versions,
    write_metadata,
)


def test_array_digest_changes_with_content_and_shape():
    a = np.zeros((2, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0] = 1

    assert array_digest(a) == array_digest(a.copy())
    assert array_digest(a) != array_digest(b)
    assert array_digest(a) != array_digest(a.reshape(3, 2))


def test_file_sha256_and_missing_file(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")

    assert file_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert file_sha256(tmp_path / "missing") is None


def test_write_metadata_is_plain_yaml(tmp_path):
    path = write_metadata(
        tmp_path / "meta.yaml",
        {"value": np.float64(1.5), "shape": (2, 3), "where": tmp_path},
    )

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["value"] == 1.5
    assert loaded["shape"] == [2, 3]
    assert loaded["where"] == str(tmp_path)
    assert "exported_at" in loaded
    assert loaded["software"]["numpy"] == software_versions()["numpy"]
