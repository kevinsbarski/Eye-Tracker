"""macOS implementation of the platform layer, through pyobjc (Quartz, AppKit, Accessibility)."""

from __future__ import annotations

import logging
import os

import AppKit
import ApplicationServices as AX
import Quartz

from eyetracker.engine.decision import InputIdle
from eyetracker.geometry import Display
from eyetracker.platform.base import WindowInfo, pick_window_on_display

log = logging.getLogger(__name__)

_MOUSE_EVENTS = (
    Quartz.kCGEventMouseMoved,
    Quartz.kCGEventLeftMouseDown,
    Quartz.kCGEventRightMouseDown,
    Quartz.kCGEventLeftMouseDragged,
    Quartz.kCGEventScrollWheel,
)
_BOUNDS_TOLERANCE = 2.0
_AX_TIMEOUT_S = 0.5
_ACCESSIBILITY_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
_CAMERA_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"


class MacPlatform:
    def __init__(self) -> None:
        self._own_pid = os.getpid()

    # --- displays and cursor ------------------------------------------------------------------

    def displays(self) -> list[Display]:
        err, ids, count = Quartz.CGGetActiveDisplayList(32, None, None)
        if err:
            raise OSError(f"CGGetActiveDisplayList failed with {err}")
        result = []
        for display_id in list(ids)[:count]:
            if Quartz.CGDisplayMirrorsDisplay(display_id) != Quartz.kCGNullDirectDisplay:
                continue  # a mirror shows the same picture; it is not a place to look
            b = Quartz.CGDisplayBounds(display_id)
            result.append(
                Display(int(display_id), b.origin.x, b.origin.y, b.size.width, b.size.height)
            )
        return result

    def cursor_position(self) -> tuple[float, float]:
        point = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return (float(point.x), float(point.y))

    def warp_cursor(self, x: float, y: float) -> None:
        Quartz.CGWarpMouseCursorPosition((x, y))
        # without this the cursor ignores the mouse for ~0.25 s after a warp
        Quartz.CGAssociateMouseAndMouseCursorPosition(True)

    def input_idle(self) -> InputIdle:
        state = Quartz.kCGEventSourceStateHIDSystemState
        since = Quartz.CGEventSourceSecondsSinceLastEventType
        key = since(state, Quartz.kCGEventKeyDown)
        mouse = min(since(state, event) for event in _MOUSE_EVENTS)
        return InputIdle(float(key), float(mouse))

    # --- windows ------------------------------------------------------------------------------

    def list_windows(self) -> list[WindowInfo]:
        options = (
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        )
        infos = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
        windows = []
        for info in infos:
            if float(info.get(Quartz.kCGWindowAlpha, 1.0)) <= 0.0:
                continue
            bounds = info.get(Quartz.kCGWindowBounds) or {}
            windows.append(
                WindowInfo(
                    pid=int(info.get(Quartz.kCGWindowOwnerPID, -1)),
                    window_id=int(info.get(Quartz.kCGWindowNumber, 0)),
                    x=float(bounds.get("X", 0)),
                    y=float(bounds.get("Y", 0)),
                    width=float(bounds.get("Width", 0)),
                    height=float(bounds.get("Height", 0)),
                    layer=int(info.get(Quartz.kCGWindowLayer, 0)),
                )
            )
        return windows

    def focus_frontmost_window_on(self, display: Display) -> bool:
        window = pick_window_on_display(self.list_windows(), display, self._own_pid)
        if window is None:
            return False
        raised = False
        if self.accessibility_trusted():
            raised = self._raise_with_accessibility(window)
        if not raised or self._frontmost_pid() != window.pid:
            app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(window.pid)
            if app is not None:
                app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)
        return True

    def _raise_with_accessibility(self, window: WindowInfo) -> bool:
        app = AX.AXUIElementCreateApplication(window.pid)
        # A hung app would otherwise block the main thread for the default ~6 s per call.
        AX.AXUIElementSetMessagingTimeout(app, _AX_TIMEOUT_S)
        err, ax_windows = AX.AXUIElementCopyAttributeValue(app, AX.kAXWindowsAttribute, None)
        if err or not ax_windows:
            return False
        for ax_window in ax_windows:
            frame = _ax_frame(ax_window)
            if frame is None or not _same_frame(frame, window):
                continue
            AX.AXUIElementPerformAction(ax_window, AX.kAXRaiseAction)
            AX.AXUIElementSetAttributeValue(ax_window, AX.kAXMainAttribute, True)
            AX.AXUIElementSetAttributeValue(app, AX.kAXFrontmostAttribute, True)
            return True
        return False

    @staticmethod
    def _frontmost_pid() -> int | None:
        app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(app.processIdentifier()) if app is not None else None

    # --- permissions and app chrome -----------------------------------------------------------

    def accessibility_trusted(self, prompt: bool = False) -> bool:
        return bool(AX.AXIsProcessTrustedWithOptions({AX.kAXTrustedCheckOptionPrompt: prompt}))

    def hide_dock_icon(self) -> None:
        AppKit.NSApplication.sharedApplication().setActivationPolicy_(
            AppKit.NSApplicationActivationPolicyAccessory
        )

    def activate_self(self) -> None:
        AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def open_accessibility_settings(self) -> None:
        _open_url(_ACCESSIBILITY_URL)

    def open_camera_settings(self) -> None:
        _open_url(_CAMERA_URL)


def _open_url(url: str) -> None:
    AppKit.NSWorkspace.sharedWorkspace().openURL_(AppKit.NSURL.URLWithString_(url))


def _ax_frame(element) -> tuple[float, float, float, float] | None:
    err_p, pos_value = AX.AXUIElementCopyAttributeValue(element, AX.kAXPositionAttribute, None)
    err_s, size_value = AX.AXUIElementCopyAttributeValue(element, AX.kAXSizeAttribute, None)
    if err_p or err_s or pos_value is None or size_value is None:
        return None
    ok_p, pos = AX.AXValueGetValue(pos_value, AX.kAXValueCGPointType, None)
    ok_s, size = AX.AXValueGetValue(size_value, AX.kAXValueCGSizeType, None)
    if not (ok_p and ok_s):
        return None
    return (float(pos.x), float(pos.y), float(size.width), float(size.height))


def _same_frame(frame: tuple[float, float, float, float], window: WindowInfo) -> bool:
    x, y, w, h = frame
    return all(
        abs(a - b) <= _BOUNDS_TOLERANCE
        for a, b in ((x, window.x), (y, window.y), (w, window.width), (h, window.height))
    )
