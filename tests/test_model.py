import json

import numpy as np
import pytest

from eyetracker.gaze.model import CalibrationFailed, GazeModel, expand, fit_with_cv
from eyetracker.geometry import Display

DISPLAYS = [
    Display(1, 0, 0, 1440, 900),
    Display(2, 1440, -180, 1920, 1080),
    Display(3, 3360, -180, 2560, 1080),
]


def _targets():
    pts = []
    for d in DISPLAYS:
        pts.append(d.center)
        for fx, fy in ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)):
            pts.append((d.x + fx * d.width, d.y + fy * d.height))
    return pts


def _dataset(curved: bool, seed: int = 0):
    """15 calibration points x 15 samples; features are a noisy function of the target point."""
    rng = np.random.default_rng(seed)
    X, Y, groups = [], [], []
    for g, (x, y) in enumerate(_targets()):
        dx = x - 2500
        bend = 0.0002 * dx * dx if curved else 0.0
        for _ in range(15):
            yaw = dx / 60 + rng.normal(0, 0.5)
            pitch = -(y - 400 - bend) / 60 + rng.normal(0, 0.5)
            iris_h = dx / 25000 + rng.normal(0, 0.01)
            iris_v = (y - 400) / 25000 + rng.normal(0, 0.01)
            roll = rng.normal(0, 1)
            face = [
                0.5 + rng.normal(0, 0.01),
                0.4 + rng.normal(0, 0.01),
                0.16 + rng.normal(0, 0.003),
            ]
            X.append([yaw, pitch, iris_h, iris_v, roll, *face])
            Y.append([x, y])
            groups.append(g)
    return np.array(X), np.array(Y), groups, DISPLAYS


@pytest.fixture
def linear_data():
    return _dataset(curved=False)


@pytest.fixture
def quadratic_data():
    return _dataset(curved=True)


def test_expand_shapes():
    assert expand(np.ones((5, 2)), 1).shape == (5, 2)
    assert expand(np.ones((5, 2)), 2).shape == (5, 5)  # + 2 squares + 1 product
    assert expand(np.ones((5, 3)), 2).shape == (5, 9)  # + 3 squares + 3 products


def test_estimate_ignores_head_position_roll_and_cross_axis_features(linear_data):
    """Calibration barely varies these, so the fit cannot learn them; they must not move the
    estimate when the user leans, shifts, tilts, or looks down at the keyboard."""
    X, Y, g, d = linear_data
    m, _ = fit_with_cv(X, Y, g, d)
    base = m.predict(X[0])
    moved = X[0].copy()
    moved[4:] += [8.0, 0.1, 0.1, 0.02]  # roll, face_x, face_y, face_scale
    assert np.allclose(m.predict(moved), base)
    looked_down = X[0].copy()
    looked_down[[1, 3]] += [-15.0, 0.05]  # pitch, iris_v
    assert m.predict(looked_down)[0] == pytest.approx(base[0])
    eyes_only = X[0].copy()
    eyes_only[2] += 0.1  # iris_h
    assert m.predict(eyes_only)[1] == pytest.approx(base[1])


def test_linear_truth_is_recovered(linear_data):
    X, Y, g, displays = linear_data
    _, rep = fit_with_cv(X, Y, g, displays)
    assert rep.error_pt < 40
    assert rep.display_accuracy > 0.97
    assert rep.n_points == 15


def test_quadratic_truth_prefers_degree_two(quadratic_data):
    _, rep = fit_with_cv(*quadratic_data)
    assert rep.degree_y == 2  # height on screen bends with how far left or right you look


def test_predict_shapes_and_roundtrip(linear_data):
    X, Y, g, d = linear_data
    m, _ = fit_with_cv(X, Y, g, d)
    assert m.predict(X[0]).shape == (2,)
    assert m.predict(X).shape == (len(X), 2)
    m2 = GazeModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.allclose(m2.predict(X), m.predict(X))


def test_constant_feature_does_not_break_fit(linear_data):
    X, Y, g, d = linear_data
    X = X.copy()
    X[:, 4] = 0.0  # e.g. roll exactly constant
    m, rep = fit_with_cv(X, Y, g, d)
    assert np.all(np.isfinite(m.predict(X)))
    assert rep.error_pt < 40


def test_too_few_points_raises():
    with pytest.raises(CalibrationFailed):
        fit_with_cv(np.zeros((4, 8)), np.zeros((4, 2)), [0, 0, 1, 1], [])


def test_model_saved_by_an_older_version_is_rejected():
    old = {
        "mean": [0] * 8,
        "std": [1] * 8,
        "degree": 1,
        "alpha": 1,
        "coef": [[0, 0]] * 8,
        "intercept": [0, 0],
    }
    with pytest.raises(ValueError):
        GazeModel.from_dict(old)
