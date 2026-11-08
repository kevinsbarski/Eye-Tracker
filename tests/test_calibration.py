import pytest

from eyetracker.config import Tuning
from eyetracker.gaze.calibration import CalibrationSession, calibration_points, fit_session
from eyetracker.gaze.model import CalibrationFailed
from eyetracker.geometry import Display
from eyetracker.vision.features import GazeFeatures

A = Display(1, 0, 0, 1440, 900)
B = Display(2, 1440, 0, 1000, 500)


def _features(yaw=0.0, pitch=0.0, openness=0.3) -> GazeFeatures:
    return GazeFeatures(yaw, pitch, 0.0, 0.0, 0.0, 0.5, 0.4, 0.16, openness)


f_ok = _features()
f_blink = _features(openness=0.05)


def test_tuning_defaults_match_spec():
    t = Tuning()
    assert (t.settle_s, t.collect_s, t.min_samples, t.inset) == (0.8, 1.2, 8, 0.15)
    assert (t.dwell_s, t.hysteresis_frac, t.away_frac) == (0.30, 0.06, 0.35)
    assert (t.mouse_grace_s, t.typing_grace_s, t.cooldown_s, t.gap_reset_s) == (1.5, 2.0, 0.6, 0.25)
    assert (t.blink_openness, t.fps) == (0.12, 15)


def test_points_order_and_inset():
    pts = calibration_points([B, A])
    assert len(pts) == 10
    assert pts[0].display_id == 1 and (pts[0].x, pts[0].y) == (720, 450)
    assert (pts[1].x, pts[1].y) == pytest.approx((216, 135))  # top-left, inset 15 %
    assert (pts[2].x, pts[2].y) == pytest.approx((1224, 135))  # top-right
    assert (pts[3].x, pts[3].y) == pytest.approx((1224, 765))  # bottom-right
    assert (pts[4].x, pts[4].y) == pytest.approx((216, 765))  # bottom-left
    assert pts[5].display_id == 2
    assert [p.index for p in pts] == list(range(10))


def test_settle_samples_ignored_and_advance():
    s = CalibrationSession(calibration_points([A, B]))
    s.start(0.0)
    assert s.in_settle(0.5)
    s.add_sample(0.5, f_ok)  # in settle → ignored
    for i in range(10):
        s.add_sample(0.8 + i * 0.1, f_ok)
    s.tick(1.9)
    assert s.current.index == 0  # collection window not over yet
    s.tick(2.01)
    assert s.current.index == 1
    X, _, groups = s.dataset()
    assert len(X) == 10 and set(groups) == {0}


def test_blinks_ignored_and_short_point_retried_then_skipped():
    s = CalibrationSession(calibration_points([A, B]))
    s.start(0.0)
    for i in range(10):
        s.add_sample(0.8 + i * 0.1, f_blink)
    s.tick(2.01)
    assert s.current.index == 0  # retried
    s.tick(4.02)
    assert s.current.index == 1 and s.skipped == [0]


def _run_full_session(points, per_point=10):
    s = CalibrationSession(points)
    t = 0.0
    s.start(t)
    while not s.done:
        p = s.current
        for i in range(per_point):
            s.add_sample(t + 0.85 + i * 0.1, _features(yaw=p.x / 50, pitch=p.y / 50))
        t += 2.01
        s.tick(t)
    return s


def test_dataset_groups_match_points():
    pts = calibration_points([A, B])
    s = _run_full_session(pts)
    X, Y, groups = s.dataset()
    assert X.shape == (100, 8) and Y.shape == (100, 2)
    assert sorted(set(groups)) == list(range(10))
    first = groups.index(3)
    assert tuple(Y[first]) == (pts[3].x, pts[3].y)
    assert s.current is None and s.done


def test_fit_session_succeeds_on_clean_data():
    s = _run_full_session(calibration_points([A, B]))
    _, report = fit_session(s, [A, B])
    assert report.display_accuracy == 1.0


def test_fit_session_fails_without_samples():
    s = CalibrationSession(calibration_points([A, B]))
    s.start(0)
    for t in range(0, 60, 2):
        s.tick(t)
    assert s.done
    with pytest.raises(CalibrationFailed):
        fit_session(s, [A, B])


R = Display(3, 2440, 0, 1920, 1080)


def test_fit_session_fails_when_a_display_is_fully_skipped():
    s = CalibrationSession(calibration_points([A, B, R]))
    t = 0.0
    s.start(t)
    while not s.done:
        p = s.current
        if p.display_id != R.id:  # the user's face is never seen while looking right
            for i in range(10):
                s.add_sample(t + 0.85 + i * 0.1, _features(yaw=p.x / 50, pitch=p.y / 50))
        t += 2.01
        s.tick(t)
        if s.current is p:  # retried: skip it again
            t += 2.01
            s.tick(t)
    with pytest.raises(CalibrationFailed, match="the right screen"):
        fit_session(s, [A, B, R])


def test_describe_display():
    from eyetracker.gaze.calibration import describe_display

    assert describe_display(A, [A]) == "the screen"
    assert describe_display(B, [B, A]) == "the right screen"
    assert describe_display(A, [B, R, A]) == "the left screen"
    assert describe_display(B, [B, R, A]) == "the middle screen"
    extra = Display(4, 4360, 0, 800, 600)
    assert describe_display(R, [A, B, R, extra]) == "screen 3 of 4 from the left"


def test_save_samples_writes_numbers_only(tmp_path):
    import json

    from eyetracker.gaze.calibration import save_samples

    s = _run_full_session(calibration_points([A, B]))
    path = tmp_path / "samples.json"
    save_samples(s, [A, B], path)
    data = json.loads(path.read_text())
    assert len(data["features"]) == len(data["targets"]) == len(data["groups"]) == 100
    assert len(data["features"][0]) == 8
    assert data["displays"][0] == [1, 0, 0, 1440, 900]
    assert data["skipped"] == []
