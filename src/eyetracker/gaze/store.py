"""Saved calibrations, one per desk layout (docked and undocked each keep their own)."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from eyetracker import paths
from eyetracker.gaze.model import CVReport, GazeModel

log = logging.getLogger(__name__)
VERSION = 1


class CalibrationStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path or paths.config_dir() / "calibrations.json"

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict) and isinstance(data.get("layouts"), dict):
                return data
            log.warning("ignoring calibration file with unexpected shape: %s", self.path)
        except FileNotFoundError:
            pass
        except (ValueError, OSError):
            log.warning("ignoring unreadable calibration file: %s", self.path)
        return {"version": VERSION, "layouts": {}}

    def load(self, signature: str) -> GazeModel | None:
        entry = self._read()["layouts"].get(signature)
        if not entry:
            return None
        try:
            return GazeModel.from_dict(entry["model"])
        except (KeyError, TypeError, ValueError):
            log.warning("ignoring broken calibration for layout %s", signature)
            return None

    def save(
        self, signature: str, model: GazeModel, report: CVReport, camera_name: str | None
    ) -> None:
        data = self._read()
        data["version"] = VERSION
        data["layouts"][signature] = {
            "model": model.to_dict(),
            "report": report.to_dict(),
            "camera": camera_name,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)
