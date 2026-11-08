"""Tuning constants and the few user settings that persist between runs."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from eyetracker import paths

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tuning:
    # switching
    dwell_s: float = 0.30
    hysteresis_frac: float = 0.06  # of the smaller "shorter side" of the two displays
    away_frac: float = 0.35  # of the nearest display's diagonal
    mouse_grace_s: float = 1.5
    typing_grace_s: float = 2.0
    cooldown_s: float = 0.6
    gap_reset_s: float = 0.25
    blink_openness: float = 0.12
    fps: int = 15
    # calibration
    settle_s: float = 0.8
    collect_s: float = 1.2
    min_samples: int = 8
    inset: float = 0.15
    min_targets_per_display: int = 3  # of the 5 dots; fewer means that display is not graded


@dataclass
class Settings:
    camera_name: str | None = None

    @classmethod
    def load(cls) -> Settings:
        path = paths.config_dir() / "settings.json"
        try:
            data = json.loads(path.read_text())
            return cls(camera_name=data.get("camera_name"))
        except FileNotFoundError:
            return cls()
        except (ValueError, AttributeError, OSError):
            log.warning("ignoring unreadable settings file %s", path)
            return cls()

    def save(self) -> None:
        path = paths.config_dir() / "settings.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)
