import json

import numpy as np

from eyetracker.config import Settings
from eyetracker.gaze.model import CVReport, GazeModel
from eyetracker.gaze.store import CalibrationStore


def _model() -> GazeModel:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 8))
    Y = X[:, :2] * 100 + 500
    return GazeModel.fit(X, Y, 1, 0.1)


REPORT = CVReport(error_pt=12.5, display_accuracy=0.98, n_points=10)


def test_save_load_roundtrip_by_signature(tmp_path):
    path = tmp_path / "calibrations.json"
    model = _model()
    CalibrationStore(path).save(
        "0,0,1440,900;1440,0,1920,1080", model, REPORT, "MacBook Air Camera"
    )
    loaded = CalibrationStore(path).load("0,0,1440,900;1440,0,1920,1080")
    x = np.ones(8)
    assert loaded is not None and np.allclose(loaded.predict(x), model.predict(x))
    assert CalibrationStore(path).load("other") is None
    data = json.loads(path.read_text())
    entry = data["layouts"]["0,0,1440,900;1440,0,1920,1080"]
    assert data["version"] == 1
    assert entry["camera"] == "MacBook Air Camera" and entry["report"]["display_accuracy"] == 0.98


def test_second_layout_keeps_first(tmp_path):
    path = tmp_path / "c.json"
    store = CalibrationStore(path)
    store.save("a", _model(), REPORT, None)
    store.save("b", _model(), REPORT, None)
    assert store.load("a") is not None and store.load("b") is not None


def test_corrupt_file_loads_as_none(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{nope")
    assert CalibrationStore(p).load("x") is None
    CalibrationStore(p).save("x", _model(), REPORT, None)  # overwrites the broken file
    assert CalibrationStore(p).load("x") is not None


def test_default_path_is_in_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EYETRACKER_CONFIG_DIR", str(tmp_path / "cfg"))
    CalibrationStore().save("s", _model(), REPORT, None)
    assert (tmp_path / "cfg" / "calibrations.json").is_file()


def test_settings_roundtrip_and_corrupt(tmp_path, monkeypatch):
    monkeypatch.setenv("EYETRACKER_CONFIG_DIR", str(tmp_path))
    assert Settings.load().camera_name is None
    Settings(camera_name="MacBook Air Camera").save()
    assert Settings.load().camera_name == "MacBook Air Camera"
    (tmp_path / "settings.json").write_text("[broken")
    assert Settings.load().camera_name is None
