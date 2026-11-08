import pytest
from face_synth import synth_landmarks

from eyetracker.vision.features import (
    FEATURE_DIMS,
    GAZE_DIMS,
    GazeFeatures,
    features_from_landmarks,
)


@pytest.mark.parametrize("axis", ["yaw", "pitch", "roll"])
@pytest.mark.parametrize("deg", [-30, -10, 10, 30])
def test_single_axis_angles(axis, deg):
    f = features_from_landmarks(synth_landmarks(**{axis: deg}), 640, 480)
    assert getattr(f, axis) == pytest.approx(deg, abs=1.5)
    for other in {"yaw", "pitch", "roll"} - {axis}:
        assert abs(getattr(f, other)) < 1.5


def test_iris_centred_and_shifted():
    centred = features_from_landmarks(synth_landmarks(), 640, 480)
    assert centred.iris_h == pytest.approx(0, abs=0.01)
    assert centred.iris_v == pytest.approx(0, abs=0.01)
    f = features_from_landmarks(synth_landmarks(iris_shift=(0.2, 0.1)), 640, 480)
    assert f.iris_h == pytest.approx(0.2, abs=0.02)
    assert f.iris_v == pytest.approx(0.1, abs=0.02)


def test_position_scale_openness():
    f = features_from_landmarks(synth_landmarks(center=(160, 360), scale=2.0), 640, 480)
    assert f.face_x == pytest.approx(160 / 640, abs=0.01)
    assert f.face_y == pytest.approx((360 - 70) / 480, abs=0.01)  # eyes sit 35*2 above centre
    assert f.face_scale == pytest.approx(180 / 640, abs=0.01)
    assert f.eye_openness == pytest.approx(14 / 30, abs=0.01)


def test_vector_order():
    f = GazeFeatures(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert list(f.vector()) == [1, 2, 4, 5, 3, 6, 7, 8]
    assert GAZE_DIMS == 4 and FEATURE_DIMS == 8
