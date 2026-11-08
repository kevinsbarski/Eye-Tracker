import pytest

from eyetracker.gaze.filters import OneEuroFilter, PointFilter


def test_first_sample_passes_through():
    assert OneEuroFilter()(0.0, 42.0) == 42.0


def test_constant_input_stays_constant():
    f = OneEuroFilter()
    assert all(f(i / 15, 100.0) == pytest.approx(100.0) for i in range(30))


def test_jitter_is_reduced():
    f = OneEuroFilter()
    out = [f(i / 15, 500 + (10 if i % 2 else -10)) for i in range(60)]
    assert max(out[30:]) - min(out[30:]) < 10


def test_fast_move_is_followed_quickly():
    f = OneEuroFilter()
    for i in range(15):
        f(i / 15, 0.0)
    out = [f(1 + i / 15, 2000.0) for i in range(1, 6)]
    assert out[-1] > 1500


def test_non_increasing_time_returns_previous_output():
    f = OneEuroFilter()
    first = f(1.0, 10.0)
    assert f(1.0, 500.0) == first


def test_reset_and_point_filter():
    p = PointFilter()
    p(0, (1, 2))
    p.reset()
    assert p(1, (5, 6)) == (5, 6)
