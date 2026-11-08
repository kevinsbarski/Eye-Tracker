"""Download the two face models once and verify their SHA-256 checksums.

This is the only part of the project that uses the network. Run it from the repo root:

    uv run python scripts/fetch_models.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "src" / "eyetracker" / "models"

LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
LANDMARKER_TASK_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
LANDMARK_MODEL = "face_landmarks_detector.tflite"
LANDMARK_MODEL_SHA256 = "c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "eyetracker-fetch-models"})
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as out:
        shutil.copyfileobj(response, out)


def _check(path: Path, expected: str, name: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise SystemExit(f"checksum mismatch for {name}: got {actual}, expected {expected}")


def _already_ok(path: Path, expected: str) -> bool:
    return path.is_file() and sha256_of(path) == expected


def fetch_all(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        landmark_path = dest / LANDMARK_MODEL
        if _already_ok(landmark_path, LANDMARK_MODEL_SHA256):
            print(f"OK (already present) {LANDMARK_MODEL}")
        else:
            task = tmp_dir / "face_landmarker.task"
            _download(LANDMARKER_TASK_URL, task)
            _check(task, LANDMARKER_TASK_SHA256, "face_landmarker.task")
            with zipfile.ZipFile(task) as archive:
                extracted = tmp_dir / LANDMARK_MODEL
                extracted.write_bytes(archive.read(LANDMARK_MODEL))
            _check(extracted, LANDMARK_MODEL_SHA256, LANDMARK_MODEL)
            shutil.move(extracted, landmark_path)
            print(f"OK {LANDMARK_MODEL}")

        yunet_path = dest / YUNET_MODEL
        if _already_ok(yunet_path, YUNET_SHA256):
            print(f"OK (already present) {YUNET_MODEL}")
        else:
            downloaded = tmp_dir / YUNET_MODEL
            _download(YUNET_URL, downloaded)
            _check(downloaded, YUNET_SHA256, YUNET_MODEL)
            shutil.move(downloaded, yunet_path)
            print(f"OK {YUNET_MODEL}")


def main() -> int:
    fetch_all(DEST)
    print(f"Models are in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
