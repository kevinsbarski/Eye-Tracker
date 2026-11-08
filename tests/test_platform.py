import sys

import pytest

from eyetracker.geometry import Display, display_at
from eyetracker.platform.base import WindowInfo, pick_window_on_display

A = Display(1, 0, 0, 1440, 900)
B = Display(2, 1440, 0, 1920, 1080)
OWN = 999

mac_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


def test_skips_menubar_layer_own_pid_and_tiny_windows():
    windows = [
        WindowInfo(10, 1, 1440, 0, 1920, 25, layer=24),  # menu bar / overlay layer
        WindowInfo(OWN, 2, 1600, 100, 800, 600, layer=0),  # our own gaze dot or calibration
        WindowInfo(11, 3, 2000, 300, 30, 30, layer=0),  # tiny helper window
        WindowInfo(12, 4, 1700, 200, 900, 700, layer=0),
    ]
    assert pick_window_on_display(windows, B, OWN).window_id == 4


def test_first_front_to_back_window_whose_centre_is_on_display_wins():
    windows = [
        WindowInfo(20, 1, 100, 100, 600, 400, layer=0),  # on A
        WindowInfo(21, 2, 1200, 100, 1000, 600, layer=0),  # straddles; centre x=1700 is on B
        WindowInfo(22, 3, 2000, 100, 600, 400, layer=0),  # on B, but further back
    ]
    assert pick_window_on_display(windows, B, OWN).window_id == 2
    assert pick_window_on_display(windows, A, OWN).window_id == 1


def test_no_window_on_display_returns_none():
    windows = [WindowInfo(20, 1, 100, 100, 600, 400, layer=0)]
    assert pick_window_on_display(windows, B, OWN) is None


@mac_only
def test_mac_displays_cursor_and_idle_are_sane():
    from eyetracker.platform.macos import MacPlatform

    p = MacPlatform()
    ds = p.displays()
    assert any(d.x == 0 and d.y == 0 for d in ds)
    assert display_at(ds, *p.cursor_position()) is not None
    idle = p.input_idle()
    assert idle.seconds_since_key >= 0 and idle.seconds_since_mouse >= 0
    assert isinstance(p.list_windows(), list)
