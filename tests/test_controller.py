import pytest
from fakes import FakeModel, FakePlatform, FakeStore, Obs, feat

from eyetracker.engine.controller import Controller
from eyetracker.geometry import Display, layout_signature

A = Display(1, 0, 0, 1440, 900)
B = Display(2, 1440, 0, 1920, 1080)
C = Display(3, 3360, 0, 2560, 1080)
HZ = 15


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(displays=(A, B), store=None, cursor=(300.0, 200.0)):
    platform = FakePlatform(displays, cursor=cursor)
    store = store or FakeStore(default=FakeModel())
    ctl = Controller(platform, store, clock=Clock())
    return ctl, platform


class Feeder:
    """Feeds observations at 15 Hz on one continuous timeline."""

    def __init__(self, ctl):
        self.ctl = ctl
        self.t = 0.0

    def look(self, point, seconds, openness=0.3):
        for _ in range(round(seconds * HZ)):
            self.ctl.on_observation(Obs(self.t, feat(*point, openness=openness)))
            self.t += 1 / HZ


@pytest.fixture(autouse=True)
def _qt(qapp):
    yield


def test_switch_warps_to_target_centre_then_focuses():
    ctl, p = make()
    switched = []
    ctl.switched.connect(switched.append)
    Feeder(ctl).look(B.center, 0.6)
    assert p.warps == [(2400, 540)]
    assert p.focused == [2]
    assert switched == [2]
    assert ctl.state == "tracking"


def test_return_restores_remembered_cursor_position():
    ctl, p = make(cursor=(300.0, 200.0))
    f = Feeder(ctl)
    f.look(B.center, 0.6)
    p.cursor = (2000.0, 300.0)  # the user moved the mouse a little on B
    f.look(A.center, 1.5)
    assert p.warps[-1] == (300.0, 200.0)
    f.look(B.center, 1.5)
    assert p.warps[-1] == (2000.0, 300.0)


def test_remembered_position_outside_target_falls_back_to_centre():
    ctl, p = make()
    f = Feeder(ctl)
    f.look(B.center, 0.6)
    p.cursor = (3300.0, 1000.0)  # bottom-right of B
    f.look(A.center, 1.5)
    smaller_b = Display(2, 1440, 0, 1280, 720)
    p.set_displays([A, smaller_b])
    ctl.refresh_displays()
    f.look(smaller_b.center, 1.5)
    assert p.warps[-1] == smaller_b.center


def test_paused_and_calibrating_never_warp_but_calibrating_publishes_features():
    ctl, p = make()
    published = []
    ctl.features_observed.connect(lambda f, t: published.append(t))
    f = Feeder(ctl)
    ctl.set_paused(True)
    assert ctl.state == "paused"
    f.look(B.center, 1.0)
    ctl.set_paused(False)
    ctl.set_calibrating(True)
    f.look(B.center, 1.0)
    assert p.warps == []
    assert len(published) >= 14


def test_blink_counts_as_no_gaze():
    ctl, p = make()
    gazes = []
    ctl.gaze_updated.connect(gazes.append)
    Feeder(ctl).look(B.center, 1.0, openness=0.05)
    assert gazes and all(g is None for g in gazes)
    assert p.warps == []


def test_focus_failure_still_warps():
    ctl, p = make()
    p.focus_raises = True
    Feeder(ctl).look(B.center, 0.6)
    assert p.warps == [(2400, 540)]


def test_layout_change_reloads_calibration_and_forgets_vanished_display():
    store = FakeStore(models={layout_signature([A, B, C]): FakeModel()})
    ctl, p = make(displays=(A, B, C), store=store)
    f = Feeder(ctl)
    f.look(C.center, 0.6)
    p.cursor = (4000.0, 200.0)  # remembered spot on C
    f.look(A.center, 1.5)
    warps_before = len(p.warps)

    p.set_displays([A, B])  # C unplugged
    ctl.refresh_displays()
    assert ctl.state == "not_calibrated"
    f.look(C.center, 1.5)
    assert len(p.warps) == warps_before

    p.set_displays([A, B, C])  # plugged back in
    ctl.refresh_displays()
    assert ctl.state == "tracking"
    f.look(C.center, 1.5)
    assert p.warps[-1] == C.center  # memory for C was dropped while it was gone


def test_single_display_state():
    ctl, p = make(displays=(A,))
    assert ctl.state == "single_display"
    Feeder(ctl).look((700, 400), 1.0)
    assert p.warps == []


def test_state_changes_are_signalled():
    ctl, _ = make()
    states = []
    ctl.state_changed.connect(states.append)
    ctl.set_paused(True)
    ctl.set_paused(False)
    ctl.set_model(None)
    assert states == ["paused", "tracking", "not_calibrated"]


def test_displays_are_refreshed_every_two_seconds():
    ctl, p = make()
    clock = ctl._clock
    p.set_displays([A])
    clock.t = 1.0
    ctl.on_observation(Obs(1.0, feat(*A.center)))
    assert ctl.state == "tracking"
    clock.t = 2.1
    ctl.on_observation(Obs(2.1, feat(*A.center)))
    assert ctl.state == "single_display"


def test_unplugged_display_is_never_warped_to_even_before_refresh():
    store = FakeStore(models={layout_signature([A, B, C]): FakeModel()})
    ctl, p = make(displays=(A, B, C), store=store)
    p.set_displays([A, B])  # C unplugged; the 2 s refresh has not come round yet
    Feeder(ctl).look(C.center, 1.0)
    assert p.warps == []
    assert p.focused == []
    assert ctl.state == "not_calibrated"


def test_stale_observations_are_dropped():
    ctl, p = make()
    clock = ctl._clock
    clock.t = 10.0
    gazes = []
    ctl.gaze_updated.connect(gazes.append)
    for i in range(15):  # a backlog from 1 s ago, e.g. after the main thread was blocked
        ctl.on_observation(Obs(9.0 + i / HZ * 0.3, feat(*B.center)))
    assert p.warps == [] and gazes == []


def test_manual_mouse_move_updates_remembered_position():
    ctl, p = make(cursor=(300.0, 200.0))
    f = Feeder(ctl)
    f.look(B.center, 0.6)  # gaze switch to B
    p.cursor = (900.0, 600.0)  # the user takes the mouse back to A by hand
    f.look(A.center, 1.0)
    p.cursor = (2000.0, 300.0)  # and then over to B by hand
    f.look(B.center, 1.0)
    f.look(A.center, 1.5)  # now a gaze switch to A
    assert p.warps[-1] == (900.0, 600.0)


def test_debug_trace_and_switch_log(caplog):
    import logging

    ctl, _ = make()
    with caplog.at_level(logging.DEBUG, logger="eyetracker.engine.controller"):
        f = Feeder(ctl)
        f.look(B.center, 0.6)
        f.look(B.center, 0.2, openness=0.05)
    text = caplog.text
    assert "trace t=" in text and "reason=dwell" in text and "face=blink" in text
    assert "switch 1 -> 2 after 0.3" in text
