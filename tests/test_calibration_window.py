from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from eyetracker.geometry import Display
from eyetracker.ui.calibration_window import CalibrationWindow
from eyetracker.vision.features import GazeFeatures

A = Display(1, 0, 0, 1440, 900)
B = Display(2, 1440, 0, 1920, 1080)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _features(x, y):
    return GazeFeatures(x / 50, y / 50, 0.0, 0.0, 0.0, 0.5, 0.4, 0.16, 0.3)


def _collect(win, *signals):
    seen = {name: [] for name in signals}
    for name in signals:
        getattr(win, name).connect(lambda *args, _n=name: seen[_n].append(args))
    return seen


def test_full_run_emits_finished(qapp):
    clock = Clock()
    win = CalibrationWindow([A, B], clock=clock)
    seen = _collect(win, "finished", "failed", "cancelled")
    win.begin()
    QTest.keyClick(win.windows[0], Qt.Key_Space)
    assert win.session is not None
    while win.session.current is not None:
        p = win.session.current
        for i in range(12):
            win.on_features(_features(p.x, p.y), clock.t + 0.85 + i * 0.1)
        clock.t += 2.01
        win.tick()
    assert len(seen["finished"]) == 1 and not seen["failed"]
    model, report = seen["finished"][0]
    assert report.display_accuracy > 0.9
    assert model.predict(_features(*B.center).vector()) is not None


def test_escape_cancels(qapp):
    win = CalibrationWindow([A, B], clock=Clock())
    seen = _collect(win, "cancelled")
    win.begin()
    assert all(w.isVisible() for w in win.windows)
    QTest.keyClick(win.windows[1], Qt.Key_Escape)
    assert len(seen["cancelled"]) == 1
    assert not any(w.isVisible() for w in win.windows)


def test_no_face_emits_failed(qapp):
    clock = Clock()
    win = CalibrationWindow([A, B], clock=clock)
    seen = _collect(win, "finished", "failed")
    win.begin()
    win.start()
    for _ in range(40):
        clock.t += 2.01
        win.tick()
    assert not seen["finished"]
    assert len(seen["failed"]) == 1
    assert "Couldn't see your face" in seen["failed"][0][0]


def test_dot_shows_only_on_the_active_display(qapp):
    clock = Clock()
    win = CalibrationWindow([A, B], clock=clock)
    win.begin()
    win.start()
    win.tick()
    first, second = win.windows
    assert first.dot is not None and second.dot is None
    assert first.dot == (720, 450)  # centre of A, in A's own coordinates


def test_result_text_mentions_skipped_dots():
    from eyetracker.ui.calibration_window import result_text

    assert "skipped" not in result_text(0.95, 80.0)
    assert "2 dots were skipped" in result_text(0.95, 80.0, skipped=2)
    assert "1 dot was skipped" in result_text(0.95, 80.0, skipped=1)
