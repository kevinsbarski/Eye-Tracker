import importlib.util
from pathlib import Path

from eyetracker import paths

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_models.py"


def _load_fetch_models():
    spec = importlib.util.spec_from_file_location("fetch_models", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_dir_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EYETRACKER_CONFIG_DIR", str(tmp_path / "cfg"))
    assert paths.config_dir() == tmp_path / "cfg"
    assert (tmp_path / "cfg").is_dir()


def test_missing_models_lists_absent_files(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "models_dir", lambda: tmp_path)
    assert set(paths.missing_models()) == {paths.LANDMARK_MODEL, paths.DETECTOR_MODEL}
    (tmp_path / paths.DETECTOR_MODEL).write_bytes(b"x")
    assert paths.missing_models() == [paths.LANDMARK_MODEL]


def test_sha256_of(tmp_path):
    fetch_models = _load_fetch_models()
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert fetch_models.sha256_of(f) == expected
