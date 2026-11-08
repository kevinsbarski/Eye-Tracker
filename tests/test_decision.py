from eyetracker.engine.decision import InputIdle, SwitchDecider
from eyetracker.geometry import Display

A = Display(1, 0, 0, 1440, 900)
B = Display(2, 1440, 0, 1920, 1080)
IDLE = InputIdle(99, 99)
HZ = 15


def run(dec, schedule, current_id=1, idle_fn=lambda t: IDLE, until=3.0, start=0.0):
    """Feed updates at 15 Hz; return (t, target) of the first switch, or None."""
    i = 0
    while (t := start + i / HZ) <= until:
        target = dec.update(t, schedule(t), [A, B], current_id, idle_fn(t))
        if target is not None:
            return t, target
        i += 1
    return None


def test_switch_after_dwell():
    t, target = run(SwitchDecider(), lambda t: B.center)
    assert target == 2
    assert 0.30 <= t < 0.30 + 1 / HZ + 1e-9


def test_short_glance_does_nothing():
    assert run(SwitchDecider(), lambda t: B.center if t < 0.2 else A.center) is None


def test_hysteresis_blocks_bezel_jitter():
    # 20 pt into B; margin is 0.06 * min(900, 1080) = 54 pt
    assert run(SwitchDecider(), lambda t: (1460, 450)) is None
    # 80 pt into B clears the margin
    assert run(SwitchDecider(), lambda t: (1520, 450))[1] == 2


def test_looking_away_blocks():
    below = (500, 900 + 0.36 * A.diagonal)  # A is the nearest display
    assert run(SwitchDecider(), lambda t: below, current_id=2) is None


def test_mouse_grace_blocks_then_dwell_restarts():
    idle = lambda t: InputIdle(99, 0.5 + t)  # noqa: E731 - mouse idle reaches 1.5 s at t = 1
    t, target = run(SwitchDecider(), lambda t: B.center, idle_fn=idle)
    assert target == 2 and t >= 1.30


def test_typing_grace_blocks():
    assert run(SwitchDecider(), lambda t: B.center, idle_fn=lambda t: InputIdle(1.9, 99)) is None


def test_cooldown_after_switch():
    dec = SwitchDecider()
    t0, _ = run(dec, lambda t: B.center)
    t1, target = run(dec, lambda t: A.center, current_id=2, start=t0 + 1 / HZ, until=t0 + 3)
    assert target == 1 and t1 >= t0 + 0.6


def test_short_gap_keeps_dwell():
    def gaze(t):
        return None if 0.15 <= t < 0.25 else B.center

    t, target = run(SwitchDecider(), gaze)
    assert target == 2 and t < 0.30 + 1 / HZ + 1e-9


def test_long_gap_restarts_dwell():
    def gaze(t):
        return None if 0.2 <= t < 0.5 else B.center

    t, target = run(SwitchDecider(), gaze)
    assert target == 2 and t >= 0.80


def test_stalled_updates_restart_dwell():
    dec = SwitchDecider()
    assert dec.update(0.0, B.center, [A, B], 1, IDLE) is None
    assert dec.update(0.2, B.center, [A, B], 1, IDLE) is None
    # nothing arrives for 0.4 s (camera hiccup); the old dwell must not count
    assert dec.update(0.6, B.center, [A, B], 1, IDLE) is None
    assert dec.update(0.85, B.center, [A, B], 1, IDLE) is None
    assert dec.update(0.9, B.center, [A, B], 1, IDLE) == 2


def test_no_switch_to_current_display():
    assert run(SwitchDecider(), lambda t: A.center) is None


def test_unknown_current_display_never_switches():
    assert run(SwitchDecider(), lambda t: B.center, current_id=99) is None


def test_last_reason_explains_each_decision():
    dec = SwitchDecider()
    dec.update(0.0, None, [A, B], 1, IDLE)
    assert dec.last_reason == "no_gaze"
    dec.update(0.0, A.center, [A, B], 1, IDLE)
    assert dec.last_reason == "same_display"
    dec.update(0.0, (1460, 450), [A, B], 1, IDLE)
    assert dec.last_reason == "hysteresis"
    dec.update(0.0, (500, 900 + 0.36 * A.diagonal), [A, B], 1, IDLE)
    assert dec.last_reason == "away"
    dec.update(0.0, B.center, [A, B], 1, InputIdle(99, 0.5))
    assert dec.last_reason == "mouse"
    dec.update(0.0, B.center, [A, B], 1, InputIdle(0.5, 99))
    assert dec.last_reason == "typing"
    dec.update(0.1, B.center, [A, B], 1, IDLE)
    assert dec.last_reason.startswith("dwell")
    dec.update(0.3, B.center, [A, B], 1, IDLE)
    assert dec.last_reason == "dwell 0.20/0.30"
    assert dec.update(0.45, B.center, [A, B], 1, IDLE) == 2
    assert dec.last_reason == "switch"
    dec.update(0.5, A.center, [A, B], 2, IDLE)
    assert dec.last_reason == "cooldown"
