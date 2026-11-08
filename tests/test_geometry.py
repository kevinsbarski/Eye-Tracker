import pytest

from eyetracker.geometry import Display, display_at, layout_signature, nearest_display

A = Display(1, 0, 0, 1440, 900)
B = Display(2, 1440, -100, 1920, 1080)


def test_contains_and_distance():
    assert A.contains(10, 10) and not A.contains(1500, 10)
    assert A.distance_to(100, 100) == 0
    assert A.distance_to(1450, 450) == pytest.approx(10)
    assert A.distance_to(1443, 904) == pytest.approx(5)


def test_center_shorter_side_diagonal():
    assert A.center == (720, 450)
    assert B.shorter_side == 1080
    assert A.diagonal == pytest.approx((1440**2 + 900**2) ** 0.5)


def test_display_at_and_nearest():
    assert display_at([A, B], 2000, 0) is B
    assert display_at([A, B], 0, 2000) is None
    assert nearest_display([A, B], 700, 950) is A


def test_layout_signature_is_order_independent():
    expected = "0,0,1440,900;1440,-100,1920,1080"
    assert layout_signature([A, B]) == layout_signature([B, A]) == expected
