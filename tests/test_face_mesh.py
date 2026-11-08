import os

import cv2
import numpy as np
import pytest

from eyetracker import paths
from eyetracker.vision.face_mesh import (
    FaceMesh,
    Roi,
    crop_transform,
    landmarks_to_frame,
    roi_from_box,
)

needs_models = pytest.mark.skipif(
    bool(paths.missing_models()), reason="run scripts/fetch_models.py"
)


def _mesh() -> FaceMesh:
    folder = paths.models_dir()
    return FaceMesh(folder / paths.LANDMARK_MODEL, folder / paths.DETECTOR_MODEL)


def test_crop_transform_maps_roi_to_input_square():
    m = crop_transform(Roi(200, 150, 100))
    assert m @ [150, 100, 1] == pytest.approx([0, 0])
    assert m @ [250, 200, 1] == pytest.approx([256, 256])


def test_landmarks_to_frame_inverts_crop():
    roi = Roi(200, 150, 100)
    pts = np.array([[0, 0, 10], [256, 256, -20]] + [[128, 128, 0]] * 476, float)
    out = landmarks_to_frame(pts, roi)
    assert out[0] == pytest.approx([150, 100, 10 * 100 / 256])
    assert out[1][:2] == pytest.approx([250, 200])


def test_roi_from_box_is_square_on_box_centre():
    assert roi_from_box(10, 20, 100, 60, 1.6) == Roi(60, 50, 160)


@needs_models
@pytest.mark.skipif(not os.environ.get("EYETRACKER_TEST_FACE"), reason="set EYETRACKER_TEST_FACE")
def test_real_face_iris_lands_between_eye_corners():
    img = cv2.imread(os.environ["EYETRACKER_TEST_FACE"])
    mesh = _mesh()
    r = mesh.process(img)
    assert r is not None and r.presence > 0.9
    lm = r.landmarks
    assert lm[33, 0] < lm[468, 0] < lm[133, 0]
    assert lm[362, 0] < lm[473, 0] < lm[263, 0]
    # second frame uses the tracked region instead of the detector and must agree
    r2 = mesh.process(img)
    assert r2 is not None
    shift = np.linalg.norm(r2.landmarks[:, :2] - lm[:, :2], axis=1)
    face_width = np.ptp(lm[:, 0])
    assert shift.mean() < 0.015 * face_width


@needs_models
def test_blank_frame_has_no_face():
    assert _mesh().process(np.zeros((480, 640, 3), np.uint8)) is None
