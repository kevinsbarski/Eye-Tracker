# Eye Tracker

Look at a monitor and your cursor and keyboard focus move there. No clicking, no reaching for the mouse.

![Animation: a webcam tracks a person's head and eyes; as they turn towards each of three screens, the cursor and keyboard focus jump to that screen](docs/demo/demo.gif)

It uses the webcam your Mac already has and runs entirely on your machine: no account, no cloud,
no network at all once it is set up.

> Status: early version for macOS. Windows and Linux are planned.

## How it works

1. The camera finds 478 points on your face, including both irises, using Google's MediaPipe
   face-landmark model running inside OpenCV.
2. From those points the app measures where your head points and where your eyes look.
3. A short calibration (look at a few dots on each screen) teaches it how *your* head and eyes map
   to *your* screens.
4. When you look at another monitor for about 0.3 seconds, the cursor jumps back to where you left
   it on that monitor, and the front window there gets keyboard focus.

It holds back on purpose:
- not while you type, and not for 2 s after;
- not while you use the mouse, and not for 1.5 s after;
- not on quick glances;
- not when you look at your phone or desk;
- not when your gaze is right at the edge between two screens.

## Requirements

- macOS 14 or newer on Apple silicon
- Two or more displays (with one display there is nothing to switch)
- [uv](https://docs.astral.sh/uv/) (installs the right Python for you)

## Setup

```bash
git clone https://github.com/kevinsbarski/Eye-Tracker.git
cd Eye-Tracker
uv sync
uv run python scripts/fetch_models.py   # one-time download of the two face models (~3 MB, checksummed)
uv run eyetracker
```

An eye icon appears in the menu bar. There is no Dock icon.

### Permissions

macOS asks for two permissions. When you run from source, they belong to your **terminal app**
(Terminal, iTerm, VS Code…), not to "Eye Tracker":

- **Camera**: so the app can see your face.
- **Accessibility**: so it can bring the right window to the front. Without it, the cursor still
  moves, but only the app is activated, not the exact window. The menu shows **Grant
  Accessibility access…** until it is allowed.

Both are in **System Settings → Privacy & Security**.

## Using it

Click the eye icon:

| Item | What it does |
|---|---|
| Status line | Tracking, Paused, Not calibrated, Only one display, or Camera unavailable |
| Pause tracking / Resume tracking | Pause turns the camera off completely (the green light goes out) |
| Calibrate… | Look at each dot as it appears, turning your head naturally. Space starts, Esc cancels. About 10 seconds per screen. |
| Camera | Pick which camera to use. The built-in camera is the default, even if an iPhone (Continuity Camera) is connected. |
| Debug → Show camera preview | See what the camera sees, with the face points drawn on |
| Debug → Show gaze dot | A small red dot follows where the app thinks you are looking: the quickest way to judge a calibration |
| Quit Eye Tracker | Quit (Ctrl+C in the terminal works too) |

Calibration is saved per desk layout, so docked and undocked setups each keep their own. Saved data
lives in `~/Library/Application Support/EyeTracker/`.

**Tips for good accuracy:** sit as you normally work, keep your face lit from the front, and
recalibrate if you move the laptop or the camera.

## Privacy

- Camera frames are analysed in memory and never saved.
- Only numbers are stored: the fitted calibration model, in plain JSON.
- Typing is detected from macOS's "seconds since the last key press" timer. The app never sees
  which keys you press.
- No synthetic clicks. Focus moves by activating the window.
- The only network access is the one-time model download script.
- The MediaPipe *Python package* is deliberately not used, because it contains a usage-logging
  client. Only the model file is used.

## Development

```bash
uv run pytest            # tests (headless; macOS-only checks skip elsewhere)
uv run ruff check .      # lint
uv run eyetracker --debug
```

To include the real-face landmark test, point `EYETRACKER_TEST_FACE` at any frontal portrait
photo.

| Folder | What lives there |
|---|---|
| `src/eyetracker/vision/` | camera, face landmarks, gaze features, the vision thread |
| `src/eyetracker/gaze/` | gaze model, smoothing, calibration, saved calibrations |
| `src/eyetracker/engine/` | the switching rules and the controller that applies them |
| `src/eyetracker/platform/` | the OS layer (macOS today; Windows and Linux go here next) |
| `src/eyetracker/ui/` | menu bar, calibration screens, preview, gaze dot |

Design notes are in `docs/superpowers/specs/`.

## Roadmap

- Windows and Linux
- A packaged, signed Mac app (`.dmg`) with start at login
- Walk-away lock, privacy hotkey, settings window
- Learning from your mouse use to refine calibration over time
- Lower CPU when nothing in front of the camera changes

## Credits

- Face landmarks: [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
  model by Google, Apache-2.0.
- Face detection: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  from the OpenCV model zoo, MIT.
- Smoothing: the One Euro filter (Casiez, Roussel & Vogel, CHI 2012).

## License

MIT. See [LICENSE](LICENSE).
