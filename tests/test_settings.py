from leaf_vein_workflow.settings import (
    DEFAULT_SETTINGS,
    PROJECT_SETTINGS_NAME,
    find_settings_path,
    load_settings,
    reset_settings_file,
    resolve_output_dir,
    write_default_settings,
)


def test_default_yaml_matches_built_in_defaults(tmp_path):
    path = write_default_settings(tmp_path / "config.yaml")

    loaded = load_settings(path)

    assert loaded.values == DEFAULT_SETTINGS
    assert loaded.warnings == []
    assert load_settings(None).values == DEFAULT_SETTINGS


def test_missing_file_uses_defaults(tmp_path):
    loaded = load_settings(tmp_path / "none.yaml")

    assert loaded.values == DEFAULT_SETTINGS
    assert loaded.warnings == []


def test_partial_file_overrides_only_given_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "calibration:\n  value: 2400\ncleanup:\n  max_hole_px: 10\n",
        encoding="utf-8",
    )

    loaded = load_settings(path)

    assert loaded.values["calibration"]["value"] == 2400.0
    assert loaded.values["calibration"]["method"] == "dpi"
    assert loaded.values["cleanup"]["max_hole_px"] == 10
    assert loaded.values["cleanup"]["small_object_px"] == 100
    assert loaded.warnings == []


def test_invalid_values_fall_back_with_warnings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "calibration:\n  method: ruler\n  value: -5\n"
        "cleanup:\n  max_hole_px: 1.5\n"
        "measurement:\n  leaf_area_unit: m2\n  colour: red\n"
        "extra: {}\n",
        encoding="utf-8",
    )

    loaded = load_settings(path)

    assert loaded.values == DEFAULT_SETTINGS
    assert len(loaded.warnings) == 6


def test_broken_yaml_uses_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("calibration: [unclosed\n", encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.values == DEFAULT_SETTINGS
    assert "Could not read" in loaded.warnings[0]


def test_reset_writes_defaults_and_keeps_backup(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("calibration:\n  value: 600\n", encoding="utf-8")

    backup = reset_settings_file(path)

    assert backup.read_text(encoding="utf-8") == "calibration:\n  value: 600\n"
    assert load_settings(path).values == DEFAULT_SETTINGS
    assert reset_settings_file(tmp_path / "new.yaml") is None


def test_project_file_takes_precedence(tmp_path, monkeypatch):
    import platformdirs

    monkeypatch.setattr(
        platformdirs, "user_config_dir", lambda *args, **kwargs: str(tmp_path / "user")
    )
    assert find_settings_path(tmp_path) == tmp_path / "user" / "config.yaml"

    write_default_settings(tmp_path / PROJECT_SETTINGS_NAME)
    assert find_settings_path(tmp_path) == tmp_path / PROJECT_SETTINGS_NAME


def test_resolve_output_dir(tmp_path):
    assert resolve_output_dir("output", tmp_path) == tmp_path / "output"
    absolute = tmp_path / "elsewhere"
    assert resolve_output_dir(str(absolute), tmp_path / "x") == absolute
