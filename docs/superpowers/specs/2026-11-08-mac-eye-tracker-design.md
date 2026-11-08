# Eye Tracker for Mac — Design

Date: 2026-11-08
Status: implemented (v0.1, macOS)

## Intent

Look at a monitor and the mouse cursor and keyboard focus move there, with no clicking and no
reaching for the mouse. It uses the webcam the Mac already has and runs entirely on the machine.

The owner uses it themselves first and publishes it publicly on their GitHub
(`kevinsbarski`). The code should be clean enough for strangers to read, run and contribute to.

## Goals (first version)

1. **Glance to switch.** If you look at another monitor for a moment, the cursor jumps there. It
   lands where you left it last time on that monitor, or at the centre the first time. Keyboard
   focus goes to the front-most window on that monitor.
2. **Calibration.** You look at dots on each monitor, about 10 seconds per monitor. The result
   is graded honestly ("picks the right monitor N % of the time") and saved per desk layout.
3. **Menu bar icon.** It shows status and offers Pause/Resume (pause turns the camera off),
   Calibrate…, a camera picker, debug views (camera preview, gaze dot), a hint when Accessibility
   access is missing, and Quit.
4. **No accidental switches.** Quick glances, typing, mouse use and looking away (phone, desk)
   never trigger a switch.

## Non-goals (for now; candidates for later versions)

- Windows and Linux. The code is structured so only `platform/` and `vision/camera.py` need
  per-OS versions later.
- Walk-away lock, shoulder guard, privacy hotkey, global hotkeys, settings window, setup
  wizard, start at login, auto-update, split-pane focus, reading-aware focus, adaptive learning
  from mouse use, motion-gated low-CPU mode.
- A packaged, signed `.app` / `.dmg`. Version 1 runs from source with `uv run eyetracker`.

## Anti-goals (failure even if it "works")

- Any network access at runtime. Only the one-time model download script touches the network.
- Saving camera frames anywhere. Frames live in memory only.
- Reading keystrokes. Typing is inferred only from the OS's "seconds since last key event"
  timer.
- Synthetic clicks. Focus changes go through window activation, never a fake click.
- Focus stolen mid-sentence, or the cursor ping-ponging at the bezel.

## Constraints

- macOS 14+ on Apple silicon. The owner's machine runs macOS 27.0.1 on a MacBook Air (arm64).
- Python 3.12, managed with `uv`. Python 3.14 is installed system-wide, but we pin 3.12 for
  dependable wheels of PySide6, pyobjc and OpenCV.
- Dependencies: `numpy`, `opencv-python-headless>=4.10`, `PySide6-Essentials>=6.7`,
  `platformdirs>=4.2`, `pyobjc-framework-Quartz`, `pyobjc-framework-Cocoa`,
  `pyobjc-framework-ApplicationServices`, `pyobjc-framework-AVFoundation` (all `>=10.3`).
- The MediaPipe Python package is **not** used. Its native library contains a usage-logging
  client that contacts Google's servers. We run Google's face-landmark network file with
  OpenCV's DNN module instead.
- Coordinates everywhere are macOS global display **points**: origin at the top-left of the main
  display, y grows downward. This matches both Quartz (`CGDisplayBounds`, `CGWarpMouseCursorPosition`)
  and Qt screen geometry.

## The owner's desk (from `system_profiler`, 2026-11-08)

- Built-in MacBook Air display (2880×1864 Retina, main).
- PL2793H external, 1920×1080.
- LG HDR WFHD ultrawide external, 2560×1080.
- Cameras: "MacBook Air Camera" (built-in) and an iPhone via Continuity Camera.

So the design must handle **three or more displays**. It must also pick the built-in camera by
default, not the iPhone.

## Decided by the owner

- Mac first; Windows and Linux later, reusing everything except the platform layer.
- Python.
- Written from scratch as our own code.
- First version = glance to switch + calibration + menu bar icon + typing/mouse hold-off.
- Personal use first, published publicly on GitHub.

## How it works

```
Webcam ─▶ VisionWorker thread ──Observation──▶ Controller (Qt main thread)
           camera → FaceMesh → features          features → GazeModel → OneEuro filter
                                                  → SwitchDecider → warp cursor + focus window
```

### Vision (`vision/`)
- **Camera.** OpenCV `VideoCapture` with the AVFoundation backend, at 640×480, processed at up to
  15 fps. Cameras are listed through AVFoundation (pyobjc). Default choice order: the
  user-chosen name, then the first built-in camera, then the first camera. The camera permission
  is requested through AVFoundation before opening.
- **FaceMesh.**
  - YuNet (`cv2.FaceDetectorYN`) finds the face when there is no tracked region. The largest
    face wins.
  - The tracked region is a square of side 1.6 × the larger side of the YuNet box. After the
    first frame, it comes from the landmarks: a square of side 1.5 × the larger side of their
    bounding box.
  - The square is cropped by affine warp to 256×256 RGB, scaled to [0,1], and run through
    MediaPipe's `face_landmarks_detector.tflite` with OpenCV DNN
    (`readNetFromTFLite`, `ENGINE_CLASSIC` when available). The network outputs `Identity`
    (478×3 landmarks in crop pixels) and `Identity_1` (a face-presence logit).
  - If sigmoid(presence) < 0.5, tracking is dropped and the next frame runs YuNet again.
  - Spike result on this Mac: OpenCV 5.0.0 with the classic engine runs the network in about
    4.4 ms. The landmarks line up on a real portrait. OpenCV 5's new engine exposes only one
    output, so the classic engine is required.
- **Features.** Computed from the landmarks (in frame pixels; z is scaled like x):
  - `yaw`, `pitch`, `roll` in degrees, from head axes built from the landmarks:
    - x axis: outer eye corner 33 → outer eye corner 263.
    - down axis: forehead 10 → chin 152.
    - forward = down × x.
    - yaw > 0 when the face turns toward image +x, pitch > 0 when it turns up,
      roll = slope angle of the 33→263 line.
  - `iris_h`: the mean over both eyes of the iris centre's fractional position from the
    image-left corner to the image-right corner, minus 0.5. Right eye: 33→133, iris 468. Left
    eye: 362→263, iris 473.
  - `iris_v`: the mean signed perpendicular offset of the iris from the corner line, divided by
    eye width (down = positive).
  - `face_x`, `face_y`: the midpoint between the eyes divided by the frame width and height.
  - `face_scale`: the distance between the outer eye corners divided by frame width.
  - `eye_openness`: the mean lid gap (159–145 and 386–374) divided by eye width. Below 0.12
    counts as a blink, and the sample is unusable.

### Gaze (`gaze/`)
- **GazeModel.**
  - One ridge regression per screen axis, on standardised features (revised 2026-11-08 after
    the first real-desk test):
    - x from yaw and iris_h;
    - y from pitch, iris_v and yaw.
  - Head position, distance and roll (face_x, face_y, face_scale, roll) are not used. A
    30-second calibration hardly varies them, so the fit learned accidental weights for them.
    On the owner's desk this moved the estimate by hundreds of points whenever they leaned or
    looked down, and pulled the LG's estimate back toward the laptop screen, delaying switches
    to it.
  - Degree 2 adds squares and pairwise products of an axis's own features.
  - Degree ∈ {1, 2} and alpha ∈ {0.01, 0.1, 1, 10, 100} are chosen per axis by
    leave-one-point-out cross-validation over calibration targets.
  - The report gives the mean error in points and the display accuracy: the share of held-out
    samples whose nearest display is the true one.
- **Smoothing.** A One Euro filter on the predicted point (min_cutoff 1.0 Hz, beta 0.005,
  d_cutoff 1.0 Hz).
- **Calibration.**
  - 5 targets per display: the centre, then the 4 points inset 15 % from the corners
    (TL, TR, BR, BL). Displays go left to right (by x, then y).
  - Per target: 0.8 s settle (samples ignored), then 1.2 s of collection. Blink samples are
    ignored. A target with fewer than 8 samples is retried once, then skipped.
  - If no usable target remains, calibration fails with a clear message.
  - Fewer than two displays means there is nothing to calibrate.
- **Store.** `calibrations.json` in the app-support directory (platformdirs, app name
  `EyeTracker`; overridable by `EYETRACKER_CONFIG_DIR`).
  - Calibrations are keyed by a layout signature: the sorted `x,y,w,h` of every display. Docking
    and undocking each keep their own calibration.
  - Writes are atomic. A corrupt file loads as "no calibration", never as a crash.

### Engine (`engine/`)
- **SwitchDecider.** A pure function of time, gaze point, displays, current display and input
  idle times. A switch fires only when all of these hold:

  | Rule | Value |
  |---|---|
  | Dwell: candidate display held continuously | 0.30 s |
  | Hysteresis: dist(gaze, current) − dist(gaze, candidate) ≥ margin | 6 % of the smaller "shorter side" of the two displays |
  | Not looking away: distance to nearest display ≤ fraction of its diagonal | 0.35 |
  | Mouse grace: no mouse movement, click or scroll for | 1.5 s |
  | Typing grace: no key press for | 2.0 s |
  | Cooldown since last switch | 0.6 s |
  | Gap: missing gaze (no face or blink) longer than this restarts dwell; shorter gaps are ignored | 0.25 s |

  Any blocking rule (mouse, typing, cooldown, away) restarts the dwell timer.
- **Controller.**
  - The current display is the one under the cursor; if the cursor is in none, the nearest.
  - On a switch: remember the cursor position on the display being left, then warp to the
    remembered position on the target (if still inside it) or its centre. Then focus the
    front-most window on the target.
  - Focus failures are logged and never block the warp.
  - The display layout is refreshed every 2 s. A changed signature reloads the matching
    calibration (or goes uncalibrated), forgets cursor memory for vanished displays, and resets
    the decider.
  - Paused or calibrating: no switching. While calibrating, features are still published.

### Platform (`platform/`)
- `Platform` protocol:
  - `displays()`, `cursor_position()`, `warp_cursor(x, y)`, `input_idle()`,
    `focus_frontmost_window_on(display)`, `accessibility_trusted(prompt)`.
- macOS implementation (pyobjc):
  - **Displays:** `CGGetActiveDisplayList`/`CGDisplayBounds`, skipping mirrored displays.
  - **Cursor:** `CGWarpMouseCursorPosition`, followed by
    `CGAssociateMouseAndMouseCursorPosition(True)`.
  - **Idle times:** `CGEventSourceSecondsSinceLastEventType` on the HID system state. Key idle
    uses `kCGEventKeyDown`. Mouse idle is the minimum over moved, left/right down, left drag and
    scroll. Warps create no HID events, so our own warps never count as "mouse use".
  - **Focus:**
    - `CGWindowListCopyWindowInfo` (on-screen only, front-to-back). Take the first window that
      is layer 0, at least 50×50, not our own process, and whose centre lies on the target
      display.
    - Raise it through Accessibility: set the app's `AXFrontmost`, run `AXRaise` on the window
      whose AX position and size match the CG bounds within 2 pt, and set it `AXMain`.
    - Fall back to `NSRunningApplication.activateWithOptions_`.
    - Without Accessibility permission, only the app is activated.
  - **Dock icon:** hidden through `NSApplicationActivationPolicyAccessory`.

### UI (`ui/`)
- **Tray** (QSystemTrayIcon, template eye icon drawn with QPainter):
  - Status line: Tracking / Paused / Not calibrated / Only one display / Camera unavailable.
  - Pause/Resume, Calibrate…, Camera ▸, Debug ▸ (camera preview, gaze dot),
    "Grant Accessibility access…" (only while missing), Quit.
- **Calibration window:**
  - One frameless, stay-on-top window per display.
  - Intro: "Look at each dot as it appears. Turn your head naturally. Space to start, Esc to
    cancel."
  - The dot is shown on the active display only.
  - Result screen for 2.5 s: display accuracy and mean error, plus advice to recalibrate if
    accuracy is below 90 %.
- **Preview:** the camera frame with landmarks (irises in red) and live feature values. The
  worker attaches frames only while it is open.
- **Gaze dot:** a 24 pt translucent, click-through, non-activating circle that follows the
  filtered gaze point.

### Models
`scripts/fetch_models.py` downloads both files once into `src/eyetracker/models/` and checks
SHA-256. The models are git-ignored, not committed.

| File | Source | SHA-256 | License |
|---|---|---|---|
| `face_landmarker.task` → extract `face_landmarks_detector.tflite` | storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task | task `64184e22…bc9ff`, tflite `c7d54204…bd43d5` | Apache-2.0 (Google) |
| `face_detection_yunet_2023mar.onnx` | github.com/opencv/opencv_zoo (models/face_detection_yunet) | `8f2383e4…552fa4` | MIT |

## Risks to verify on the real Mac

1. **Camera order.** The AVFoundation camera order must match OpenCV's index order. The iPhone
   Continuity Camera must not be picked by default.
2. **Key idle timer.** `kCGEventKeyDown` idle may need Input Monitoring permission on macOS 27.
   Fallback: treat "any input while the cursor didn't move" as typing.
3. **Activation.** Activating another app from a background (accessory) app may be refused under
   macOS cooperative activation. The AX route is primary for that reason.
4. **Calibration windows.** Must cover each display without creating native-fullscreen Spaces.
5. **Accuracy.** Telling three displays apart from a built-in camera may need more targets per
   display than five. Tune after the first real calibration.

## Left to the builder

Internal helper names, logging wording, icon artwork, exact Qt widget composition, and any
constant not pinned above.
