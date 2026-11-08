# Mac Eye Tracker MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A menu-bar app for macOS that moves the cursor and keyboard focus to the monitor you look at, using the built-in webcam, entirely offline.

**Architecture:** A vision thread reads the camera, finds 478 face landmarks with MediaPipe's network running on OpenCV DNN, and emits gaze features. The Qt main-thread controller maps features through a per-user ridge-regression model and a One Euro filter, then a pure switch decider. On a switch, the controller warps the cursor and focuses the front-most window on the target display through a thin macOS platform layer (pyobjc).

**Tech Stack:** Python 3.12 (uv), numpy, opencv-python-headless (5.0 on this Mac), PySide6-Essentials, platformdirs, pyobjc (Quartz, Cocoa, ApplicationServices, AVFoundation), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-08-mac-eye-tracker-design.md`

## Global Constraints

- Python `3.12` (`.python-version`), `requires-python = ">=3.12,<3.14"`; run everything through `uv run`.
- Package name `eyetracker`, src layout `src/eyetracker/`; console script `eyetracker = "eyetracker.app:main"`.
- License MIT, `Copyright (c) 2026 kevinsbarski`.
- No network access in `src/` (only `scripts/fetch_models.py`). No `imwrite`/`VideoWriter`/`imencode` anywhere. No keyboard hooks; no synthetic clicks.
- All coordinates are macOS global display points, origin at the main display's top-left, y down.
- Never import the `mediapipe` package.
- Models live in `src/eyetracker/models/` (git-ignored); SHA-256: task `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`, extracted tflite `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5`, YuNet `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- Tuning defaults (spec, verbatim): dwell 0.30 s, hysteresis 0.06 × shorter side, away 0.35 × diagonal, mouse grace 1.5 s, typing grace 2.0 s, cooldown 0.6 s, gap reset 0.25 s, blink openness < 0.12, 15 fps, One Euro (min_cutoff 1.0, beta 0.005, d_cutoff 1.0), calibration settle 0.8 s / collect 1.2 s / min 8 samples / inset 0.15.
- Tests run headless: `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` and points `EYETRACKER_CONFIG_DIR` at a tmp dir. Mac-only tests use `pytest.mark.skipif(sys.platform != "darwin")`.
- Commits are local only; nothing is pushed to GitHub until the owner says so.

## Review Focus

1. **Continuity Camera iPhone listed first.** Expect the built-in "MacBook Air Camera" to be chosen when no camera is saved. Test: `test_choose_camera_prefers_builtin_over_iphone` (Task 10).
2. **A monitor unplugged while running.** Expect no warp to a vanished display, the matching calibration for the new layout (or "Not calibrated"), and cursor memory for the gone display dropped. Test: `test_layout_change_reloads_calibration_and_forgets_vanished_display` (Task 9).
3. **Blink or face lost mid-dwell.** Expect a gap shorter than 0.25 s to be ignored and a longer gap to restart dwell, so a stale gaze never fires. Tests: `test_short_gap_keeps_dwell`, `test_long_gap_restarts_dwell` (Task 8).
4. **Accessibility not granted, or focus throws.** Expect the cursor to still move, no crash, and the tray to offer "Grant Accessibility access…". Test: `test_focus_failure_still_warps` (Task 9).
5. **Camera busy (FaceTime/Zoom) or permission denied.** Expect status "Camera unavailable", a retry every 5 s, and no crash. Test: `test_step_reports_unavailable_when_camera_will_not_open` (Task 10).

---

### Task 1: Project scaffold and display geometry

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `LICENSE`, `src/eyetracker/__init__.py` (`__version__ = "0.1.0"`), `src/eyetracker/models/.gitkeep`, `src/eyetracker/geometry.py`, `tests/conftest.py`, `tests/test_geometry.py`

**Interfaces:**
- Produces: `Display(id: int, x: float, y: float, width: float, height: float)` (frozen dataclass) with properties `center -> tuple[float, float]`, `shorter_side -> float`, `diagonal -> float`, and methods `contains(px, py) -> bool`, `distance_to(px, py) -> float` (0 inside, else Euclidean distance to the rectangle). Free functions `display_at(displays, px, py) -> Display | None`, `nearest_display(displays, px, py) -> Display` and `layout_signature(displays) -> str` (sorted `"x,y,w,h"` of rounded ints joined by `";"`).

- [ ] **Step 1: Scaffold.** `git init`; write `pyproject.toml` (hatchling, deps from Global Constraints/spec, `[dependency-groups] dev = ["pytest>=8.2", "ruff>=0.6"]`, `[tool.pytest.ini_options] testpaths = ["tests"]`, ruff line-length 100). Write `.gitignore` (`.venv/`, `__pycache__/`, `src/eyetracker/models/*` except `.gitkeep`, `.pytest_cache/`, `.ruff_cache/`, `dist/`). Add `LICENSE` and `.python-version`. In `conftest.py`, set `QT_QPA_PLATFORM=offscreen` and add an autouse fixture that sets `EYETRACKER_CONFIG_DIR` to `tmp_path`. Run `uv sync`. Expected: resolves and installs without error.
- [ ] **Step 2: Write failing tests** in `tests/test_geometry.py`:

```python
A = Display(1, 0, 0, 1440, 900); B = Display(2, 1440, -100, 1920, 1080)
def test_contains_and_distance():
    assert A.contains(10, 10) and not A.contains(1500, 10)
    assert A.distance_to(100, 100) == 0
    assert A.distance_to(1450, 450) == pytest.approx(10)
    assert A.distance_to(1443, 904) == pytest.approx(5)
def test_center_shorter_side_diagonal():
    assert A.center == (720, 450); assert B.shorter_side == 1080
    assert A.diagonal == pytest.approx((1440**2 + 900**2) ** 0.5)
def test_display_at_and_nearest():
    assert display_at([A, B], 2000, 0) is B and display_at([A, B], 0, 2000) is None
    assert nearest_display([A, B], 700, 950) is A
def test_layout_signature_is_order_independent():
    assert layout_signature([A, B]) == layout_signature([B, A]) == "0,0,1440,900;1440,-100,1920,1080"
```

- [ ] **Step 3:** Run `uv run pytest tests/test_geometry.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 4:** Implement `src/eyetracker/geometry.py` per Interfaces. Sort the signature by (x, y).
- [ ] **Step 5:** Run `uv run pytest -v`. Expected: 4 passed.
- [ ] **Step 6: Commit** `feat: project scaffold and display geometry`.

---

### Task 2: Model download script and paths

**Files:**
- Create: `scripts/fetch_models.py`, `src/eyetracker/paths.py`, `tests/test_paths_and_fetch.py`

**Interfaces:**
- Produces: `paths.models_dir() -> Path` (the package's `models/` folder); `paths.config_dir() -> Path` (`$EYETRACKER_CONFIG_DIR` or `platformdirs.user_data_dir("EyeTracker", appauthor=False)`, created on access); constants `paths.LANDMARK_MODEL = "face_landmarks_detector.tflite"`, `paths.DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"`; `paths.missing_models() -> list[str]`. Script functions `sha256_of(path: Path) -> str` and `fetch_all(dest: Path) -> None`, plus `main() -> int`.

- [ ] **Step 1: Write failing tests:**

```python
def test_config_dir_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EYETRACKER_CONFIG_DIR", str(tmp_path / "cfg"))
    assert paths.config_dir() == tmp_path / "cfg" and (tmp_path / "cfg").is_dir()
def test_missing_models_lists_absent_files(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "models_dir", lambda: tmp_path)
    assert set(paths.missing_models()) == {paths.LANDMARK_MODEL, paths.DETECTOR_MODEL}
def test_sha256_of(tmp_path):
    f = tmp_path / "x"; f.write_bytes(b"abc")
    assert fetch_models.sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
```

Load the script in the test with `importlib.util.spec_from_file_location`.

- [ ] **Step 2:** Run `uv run pytest tests/test_paths_and_fetch.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `paths.py`, then `scripts/fetch_models.py`:
  - Download with `urllib.request` to a temp file and verify the hash before moving it into place. Extract only `face_landmarks_detector.tflite` from the `.task` zip and verify its hash.
  - Skip files whose hash already matches. On a mismatch, raise `SystemExit("checksum mismatch for <name>")`.
  - URLs: the face landmarker URL from the spec table, and `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`.
- [ ] **Step 4:** Run the tests (expected: 3 passed). Then run `uv run python scripts/fetch_models.py` (expected: prints the two files as OK) and `ls src/eyetracker/models` (expected: both files present).
- [ ] **Step 5: Commit** `feat: checksum-verified model download script`.

---

### Task 3: Gaze features from landmarks

**Files:**
- Create: `src/eyetracker/vision/__init__.py`, `src/eyetracker/vision/features.py`, `tests/face_synth.py`, `tests/test_features.py`

**Interfaces:**
- Produces: `GazeFeatures` (frozen dataclass: `yaw, pitch, roll, iris_h, iris_v, face_x, face_y, face_scale, eye_openness`, all floats) with `vector() -> np.ndarray` of shape (8,), order `[yaw, pitch, iris_h, iris_v, roll, face_x, face_y, face_scale]`; module constants `GAZE_DIMS = 4` and `FEATURE_DIMS = 8`; `features_from_landmarks(lm: np.ndarray (478,3), frame_w: int, frame_h: int) -> GazeFeatures`. Landmark index constants as in the spec.
- Test helper `tests/face_synth.py`: `synth_landmarks(yaw=0, pitch=0, roll=0, iris_shift=(0.0, 0.0), center=(320, 240), scale=1.0) -> np.ndarray (478,3)`. It places the canonical points below in camera coordinates (x right, y down, z away from the camera), rotates them about the face origin (yaw about y so the face turns toward +x, pitch about x so the face turns up, roll about z so 263 moves down), scales, and translates. Every other row is zeros.

  | idx | point | idx | point |
  |---|---|---|---|
  | 33 | (-45,-35,10) | 263 | (45,-35,10) |
  | 133 | (-15,-35,5) | 362 | (15,-35,5) |
  | 159/145 | (-30,-42,3)/(-30,-28,3) | 386/374 | (30,-42,3)/(30,-28,3) |
  | 468 | (-30,-35,3) + shift | 473 | (30,-35,3) + shift |
  | 1 | (0,5,-30) | 10 / 152 | (0,-90,5) / (0,85,5) |

  `iris_shift` is a fraction of eye width (30): x moves along the eye line, y moves perpendicular to it.

- [ ] **Step 1: Write failing tests:**

```python
@pytest.mark.parametrize("axis", ["yaw", "pitch", "roll"])
@pytest.mark.parametrize("deg", [-30, -10, 10, 30])
def test_single_axis_angles(axis, deg):
    f = features_from_landmarks(synth_landmarks(**{axis: deg}), 640, 480)
    assert getattr(f, axis) == pytest.approx(deg, abs=1.5)
    for other in {"yaw", "pitch", "roll"} - {axis}:
        assert abs(getattr(f, other)) < 1.5
def test_iris_centred_and_shifted():
    assert features_from_landmarks(synth_landmarks(), 640, 480).iris_h == pytest.approx(0, abs=0.01)
    f = features_from_landmarks(synth_landmarks(iris_shift=(0.2, 0.1)), 640, 480)
    assert f.iris_h == pytest.approx(0.2, abs=0.02) and f.iris_v == pytest.approx(0.1, abs=0.02)
def test_position_scale_openness():
    f = features_from_landmarks(synth_landmarks(center=(160, 360), scale=2.0), 640, 480)
    assert f.face_x == pytest.approx(160 / 640, abs=0.01)
    assert f.face_y == pytest.approx((360 - 70) / 480, abs=0.01)   # eyes sit 35*2 above centre
    assert f.face_scale == pytest.approx(180 / 640, abs=0.01)
    assert f.eye_openness == pytest.approx(14 / 30, abs=0.01)
def test_vector_order():
    f = GazeFeatures(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert list(f.vector()) == [1, 2, 4, 5, 3, 6, 7, 8]
```

- [ ] **Step 2:** Run `uv run pytest tests/test_features.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `features_from_landmarks`:
  - Orthonormalise x = 33→263 and down = 10→152.
  - forward = normalise(down × x).
  - yaw = atan2(fwd.x, −fwd.z); pitch = atan2(−fwd.y, hypot(fwd.x, fwd.z)); roll = atan2(x.y, x.x), all in degrees.
  - iris and eye_openness are 2-D in image coordinates, per the spec.
- [ ] **Step 4:** Run the tests. Expected: all pass.
- [ ] **Step 5: Commit** `feat: gaze features from face landmarks`.

---

### Task 4: FaceMesh (detector + landmark network)

**Files:**
- Create: `src/eyetracker/vision/face_mesh.py`, `tests/test_face_mesh.py`

**Interfaces:**
- Consumes: `paths.models_dir()`, `LANDMARK_MODEL`, `DETECTOR_MODEL` (Task 2).
- Produces:
  - `Roi(cx: float, cy: float, side: float)` (frozen dataclass).
  - `roi_from_box(x, y, w, h, scale) -> Roi`.
  - `crop_transform(roi: Roi, size: int = 256) -> np.ndarray` (2×3 affine).
  - `landmarks_to_frame(pts: np.ndarray (478,3), roi: Roi, size: int = 256) -> np.ndarray` (x and y mapped back; z × side/size).
  - `FaceResult(landmarks: np.ndarray, presence: float)`.
  - `class FaceMesh(landmark_path: Path, detector_path: Path)` with `process(frame_bgr) -> FaceResult | None` and `reset() -> None`. Constants: `DETECTION_SCALE = 1.6`, `TRACKING_SCALE = 1.5`, `PRESENCE_THRESHOLD = 0.5`.

- [ ] **Step 1: Write failing tests:**

```python
def test_crop_transform_maps_roi_to_input_square():
    m = crop_transform(Roi(200, 150, 100))
    assert m @ [150, 100, 1] == pytest.approx([0, 0]); assert m @ [250, 200, 1] == pytest.approx([256, 256])
def test_landmarks_to_frame_inverts_crop():
    roi = Roi(200, 150, 100); pts = np.array([[0, 0, 10], [256, 256, -20]] + [[128, 128, 0]] * 476, float)
    out = landmarks_to_frame(pts, roi)
    assert out[0] == pytest.approx([150, 100, 10 * 100 / 256]); assert out[1][:2] == pytest.approx([250, 200])
def test_roi_from_box_is_square_on_box_centre():
    assert roi_from_box(10, 20, 100, 60, 1.6) == Roi(60, 50, 160)
@pytest.mark.skipif(not os.environ.get("EYETRACKER_TEST_FACE") or paths.missing_models(), reason="needs models and a face photo")
def test_real_face_iris_lands_between_eye_corners():
    img = cv2.imread(os.environ["EYETRACKER_TEST_FACE"])
    r = FaceMesh(paths.models_dir() / paths.LANDMARK_MODEL, paths.models_dir() / paths.DETECTOR_MODEL).process(img)
    assert r is not None and r.presence > 0.9
    lm = r.landmarks; assert lm[33, 0] < lm[468, 0] < lm[133, 0] and lm[362, 0] < lm[473, 0] < lm[263, 0]
def test_blank_frame_has_no_face():  # skip when models are missing
    assert FaceMesh(...).process(np.zeros((480, 640, 3), np.uint8)) is None
```

- [ ] **Step 2:** Run `uv run pytest tests/test_face_mesh.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `FaceMesh`:
  - Load YuNet with `cv2.FaceDetectorYN.create(path, "", (w, h), 0.7, 0.3, 5000)` and call `setInputSize` per frame size.
  - Load the network with `cv2.dnn.readNetFromTFLite(path, engine=cv2.dnn.ENGINE_CLASSIC)` when `hasattr(cv2.dnn, "ENGINE_CLASSIC")`, else without `engine`.
  - Build the input with `warpAffine` (constant border) and `blobFromImage(crop, 1/255, (256, 256), swapRB=True)`, then `forward(["Identity", "Identity_1"])`.
  - Presence = sigmoid(logit). Below threshold: drop the ROI and return None. Otherwise, set the next ROI from the landmark bounding box with `TRACKING_SCALE`.
- [ ] **Step 4:** Run `EYETRACKER_TEST_FACE=<a portrait jpg> uv run pytest tests/test_face_mesh.py -v`. Expected: all pass, real-face test included. (The spike photo from the session scratchpad works.)
- [ ] **Step 5: Commit** `feat: face landmarks with OpenCV DNN`.

---

### Task 5: One Euro filter

**Files:**
- Create: `src/eyetracker/gaze/__init__.py`, `src/eyetracker/gaze/filters.py`, `tests/test_filters.py`

**Interfaces:**
- Produces: `OneEuroFilter(min_cutoff=1.0, beta=0.005, d_cutoff=1.0)` with `__call__(t: float, x: float) -> float` and `reset()`; `PointFilter` with the same parameters, `__call__(t, (x, y)) -> tuple[float, float]` and `reset()` (two independent filters).

- [ ] **Step 1: Write failing tests:**

```python
def test_first_sample_passes_through():
    assert OneEuroFilter()(0.0, 42.0) == 42.0
def test_constant_input_stays_constant():
    f = OneEuroFilter(); assert all(f(i / 15, 100.0) == pytest.approx(100.0) for i in range(30))
def test_jitter_is_reduced():
    f = OneEuroFilter(); out = [f(i / 15, 500 + (10 if i % 2 else -10)) for i in range(60)]
    assert max(out[30:]) - min(out[30:]) < 10
def test_fast_move_is_followed_quickly():
    f = OneEuroFilter(); [f(i / 15, 0.0) for i in range(15)]
    out = [f(1 + i / 15, 2000.0) for i in range(1, 6)]
    assert out[-1] > 1500
def test_reset_and_point_filter():
    p = PointFilter(); p(0, (1, 2)); p.reset(); assert p(1, (5, 6)) == (5, 6)
```

- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement the standard One Euro algorithm. When dt ≤ 0, return the previous output. **Step 4:** Run, expect PASS.
- [ ] **Step 5: Commit** `feat: One Euro smoothing filter`.

---

### Task 6: Gaze model with cross-validation

**Files:**
- Create: `src/eyetracker/gaze/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `GAZE_DIMS`, `FEATURE_DIMS` (Task 3); `Display`, `nearest_display` (Task 1).
- Produces:
  - `expand(z: np.ndarray (n,8), degree: int) -> np.ndarray`. Degree 1 is `z`. Degree 2 appends the 4 squares and 6 pairwise products of columns 0–3 only.
  - `class GazeModel` with:
    - `fit(X (n,8), Y (n,2), degree, alpha) -> GazeModel` (classmethod). Standardise X; ridge with an unpenalised intercept. A zero std becomes 1.
    - `predict(x) -> np.ndarray`: (8,) → (2,), (n,8) → (n,2).
    - `to_dict() -> dict` (JSON-safe lists) and `from_dict(d) -> GazeModel`.
  - `@dataclass CVReport(degree: int, alpha: float, error_pt: float, display_accuracy: float, n_points: int)`.
  - `fit_with_cv(X, Y, groups, displays, degrees=(1, 2), alphas=(0.01, 0.1, 1, 10, 100)) -> tuple[GazeModel, CVReport]`: leave-one-group-out, choose minimum mean error, refit on all data.
  - `class CalibrationFailed(Exception)`, raised when fewer than 3 groups exist.

- [ ] **Step 1: Write failing tests.** A fixture builds synthetic data: 15 groups (5 per display over 3 displays, `Display(1,0,0,1440,900)`, `Display(2,1440,-180,1920,1080)`, `Display(3,3360,-180,2560,1080)`), 15 samples each, features ~ N(0, 1) per group mean plus noise 0.05.

```python
def test_expand_shapes():
    z = np.ones((5, 8)); assert expand(z, 1).shape == (5, 8) and expand(z, 2).shape == (5, 18)
def test_position_features_never_enter_nonlinear_terms():
    z = np.zeros((1, 8)); z[0, 4:] = 3.0
    assert np.all(expand(z, 2)[0, 8:] == 0)
def test_linear_truth_is_recovered(linear_data):
    X, Y, g, displays = linear_data
    model, rep = fit_with_cv(X, Y, g, displays)
    assert rep.error_pt < 40 and rep.display_accuracy > 0.97
def test_quadratic_truth_prefers_degree_two(quadratic_data):
    _, rep = fit_with_cv(*quadratic_data); assert rep.degree == 2
def test_predict_shapes_and_roundtrip(linear_data):
    X, Y, g, d = linear_data; m, _ = fit_with_cv(X, Y, g, d)
    assert m.predict(X[0]).shape == (2,) and m.predict(X).shape == (len(X), 2)
    m2 = GazeModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.allclose(m2.predict(X), m.predict(X))
def test_too_few_points_raises():
    with pytest.raises(CalibrationFailed): fit_with_cv(np.zeros((4, 8)), np.zeros((4, 2)), [0, 0, 1, 1], [])
```

- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement with numpy only (`np.linalg.solve` on `(AᵀA + αI)`, intercept by centring). **Step 4:** Run, expect PASS.
- [ ] **Step 5: Commit** `feat: ridge gaze model with leave-one-point-out CV`.

---

### Task 7: Calibration session and calibration store

**Files:**
- Create: `src/eyetracker/config.py`, `src/eyetracker/gaze/calibration.py`, `src/eyetracker/gaze/store.py`, `tests/test_calibration.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `Display`, `layout_signature` (Task 1); `GazeFeatures` (Task 3); `GazeModel`, `CVReport`, `fit_with_cv`, `CalibrationFailed` (Task 6); `paths.config_dir()` (Task 2).
- Produces:
  - `config.Tuning`: a frozen dataclass of every Global Constraints tuning value, with field names `dwell_s, hysteresis_frac, away_frac, mouse_grace_s, typing_grace_s, cooldown_s, gap_reset_s, blink_openness, fps, settle_s, collect_s, min_samples, inset`.
  - `config.Settings(camera_name: str | None = None)` with `load() -> Settings` and `save()`, stored as `config_dir()/settings.json`. A corrupt file loads the defaults.
  - `CalPoint(index: int, display_id: int, x: float, y: float)`.
  - `calibration_points(displays, inset=0.15) -> list[CalPoint]`.
  - `class CalibrationSession(points, tuning=Tuning())`:
    - `start(now)`, `current -> CalPoint | None`, `in_settle(now) -> bool`.
    - `add_sample(now, f: GazeFeatures)`.
    - `tick(now) -> None` (advances, retries once, then skips).
    - `done -> bool`, `skipped -> list[int]`.
    - `dataset() -> tuple[np.ndarray, np.ndarray, list[int]]`.
  - `fit_session(session, displays) -> tuple[GazeModel, CVReport]` (raises `CalibrationFailed`).
  - `class CalibrationStore(path: Path | None = None)` (default `config_dir()/calibrations.json`) with `load(signature: str) -> GazeModel | None` and `save(signature, model, report, camera_name)`.

- [ ] **Step 1: Write failing tests** (`f_ok` = GazeFeatures with eye_openness 0.3; `f_blink` with 0.05):

```python
def test_points_order_and_inset():
    pts = calibration_points([Display(2, 1440, 0, 1000, 500), Display(1, 0, 0, 1440, 900)])
    assert len(pts) == 10 and pts[0].display_id == 1 and (pts[0].x, pts[0].y) == (720, 450)
    assert (pts[1].x, pts[1].y) == pytest.approx((216, 135))        # TL inset 15 %
    assert [p.index for p in pts] == list(range(10))
def test_settle_samples_ignored_and_advance():
    s = CalibrationSession(calibration_points([A, B])); s.start(0.0)
    s.add_sample(0.5, f_ok)                       # in settle → ignored
    for i in range(10): s.add_sample(0.8 + i * 0.1, f_ok)
    s.tick(2.01); assert s.current.index == 1
def test_blinks_ignored_and_short_point_retried_then_skipped():
    s = CalibrationSession(calibration_points([A, B])); s.start(0.0)
    for i in range(10): s.add_sample(0.8 + i * 0.1, f_blink)
    s.tick(2.01); assert s.current.index == 0   # retried
    s.tick(4.02); assert s.current.index == 1 and s.skipped == [0]
def test_dataset_groups_match_points(): ...   # 10 samples per point → dataset shapes (100, 8), (100, 2), groups 0..9
def test_fit_session_fails_without_samples():
    s = CalibrationSession(calibration_points([A, B])); s.start(0); [s.tick(t) for t in range(0, 60, 2)]
    with pytest.raises(CalibrationFailed): fit_session(s, [A, B])
# test_store.py
def test_save_load_roundtrip_by_signature(tmp_path): ...  # load(sig) predicts the same as saved; load("other") is None
def test_corrupt_file_loads_as_none(tmp_path):
    p = tmp_path / "c.json"; p.write_text("{nope"); assert CalibrationStore(p).load("x") is None
def test_settings_roundtrip_and_corrupt(): ...
```

- [ ] **Step 2:** Run both files, expect FAIL. **Step 3:** Implement. Store writes go through `tmp` + `os.replace`. JSON shape: `{"version": 1, "layouts": {sig: {"model": {...}, "report": {...}, "camera": str|null, "created": iso}}}`. **Step 4:** Run, expect PASS.
- [ ] **Step 5: Commit** `feat: calibration session, settings and per-layout store`.

---

### Task 8: Switch decider

**Files:**
- Create: `src/eyetracker/engine/__init__.py`, `src/eyetracker/engine/decision.py`, `tests/test_decision.py`

**Interfaces:**
- Consumes: `Display`, `nearest_display` (Task 1); `Tuning` (Task 7).
- Produces:
  - `InputIdle(seconds_since_key: float, seconds_since_mouse: float)` (frozen dataclass, defined here; platform re-exports it).
  - `class SwitchDecider(tuning=Tuning())` with `update(now: float, gaze: tuple[float, float] | None, displays: list[Display], current_id: int, idle: InputIdle) -> int | None` (target display id when a switch fires) and `reset()`.

- [ ] **Step 1: Write failing tests.** Displays `A=Display(1,0,0,1440,900)`, `B=Display(2,1440,0,1920,1080)`. `IDLE = InputIdle(99, 99)`. Helper `run(dec, points_by_time)` feeds updates at 15 Hz and returns `(t, target)` of the first non-None.

```python
def test_switch_after_dwell():            # gaze at B centre from t=0 → fires at first t ≥ 0.30, never before
def test_short_glance_does_nothing():     # 0.2 s on B then back to A → None throughout
def test_hysteresis_blocks_bezel_jitter():# gaze at (1460, 450): only 20 pt into B, margin 0.06*900=54 → no switch
def test_looking_away_blocks():           # gaze at (500, 900 + 0.36*diag(A)) → no switch (A is nearest)
def test_mouse_grace_blocks_then_dwell_restarts():  # idle mouse 1.0 → none; mouse idle ≥1.5 from t=1 → fires at ≥1.30
def test_typing_grace_blocks():           # seconds_since_key=1.9 → none
def test_cooldown_after_switch():         # after switch to B at t0, gaze back on A: no switch before t0+0.6
def test_short_gap_keeps_dwell():         # B at 0–0.15, None at 0.15–0.25, B after → fires at first t ≥ 0.30 (≈0.333 at 15 Hz)
def test_long_gap_restarts_dwell():       # B 0–0.2, None 0.2–0.5, B after → no fire before 0.80
def test_no_switch_to_current_display():  # gaze inside A while current is A → None
```

- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement per the spec's rule table. Any blocking rule clears the candidate start time. A gap is "time since last non-None gaze". **Step 4:** Run, expect PASS.
- [ ] **Step 5: Commit** `feat: switch decider with dwell, hysteresis and input guards`.

---

### Task 9: Platform layer and controller

**Files:**
- Create: `src/eyetracker/platform/__init__.py`, `src/eyetracker/platform/base.py`, `src/eyetracker/platform/macos.py`, `src/eyetracker/engine/controller.py`, `tests/fakes.py`, `tests/test_platform.py`, `tests/test_controller.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 5, 6, 7 and 8 (`Display`, `display_at`, `nearest_display`, `layout_signature`, `GazeFeatures`, `PointFilter`, `GazeModel`, `CalibrationStore`, `Tuning`, `SwitchDecider`, `InputIdle`).
- Produces:
  - `base.WindowInfo(pid: int, window_id: int, x: float, y: float, width: float, height: float, layer: int)`.
  - `base.Platform` (Protocol): `displays() -> list[Display]`, `cursor_position() -> tuple[float, float]`, `warp_cursor(x, y) -> None`, `input_idle() -> InputIdle`, `focus_frontmost_window_on(display) -> bool`, `accessibility_trusted(prompt: bool = False) -> bool`.
  - `base.pick_window_on_display(windows: list[WindowInfo], display, own_pid) -> WindowInfo | None`.
  - `macos.MacPlatform(Platform)` adds `hide_dock_icon()`, `open_accessibility_settings()`, `open_camera_settings()`, `activate_self()`.
  - `controller.Controller(QObject)`:
    - Constructor `(platform, store: CalibrationStore, tuning=Tuning(), clock=time.monotonic)`.
    - Signals: `gaze_updated(object)` (point or None), `features_observed(object, float)`, `switched(int)`, `state_changed(str)` (one of `"tracking"`, `"paused"`, `"not_calibrated"`, `"single_display"`).
    - Methods: `on_observation(obs)` (slot), `refresh_displays()`, `set_paused(bool)`, `set_calibrating(bool)`, `set_model(model)`, and properties `displays`, `signature`, `state`.
  - `tests/fakes.py`: `FakePlatform` records warps and focus calls, with settable cursor, idle and displays and `focus_raises: bool`. `FakeModel.predict(v)` returns `v[:2]`. `Obs(t, features)`.

- [ ] **Step 1: Write failing tests:**

```python
# test_platform.py — pick_window_on_display
def test_skips_menubar_layer_own_pid_and_tiny_windows(): ...
def test_first_front_to_back_window_whose_centre_is_on_display_wins(): ...
@mac_only
def test_mac_displays_cursor_and_idle_are_sane():
    p = MacPlatform(); ds = p.displays(); assert any(d.x == 0 and d.y == 0 for d in ds)
    assert display_at(ds, *p.cursor_position()) is not None
    i = p.input_idle(); assert i.seconds_since_key >= 0 and i.seconds_since_mouse >= 0
# test_controller.py (A=(0,0,1440,900), B=(1440,0,1920,1080); features whose yaw,pitch = gaze point)
def test_switch_warps_to_target_centre_then_focuses(): ...     # warps == [(2400, 540)], focused == [2]
def test_return_restores_remembered_cursor_position(): ...
def test_remembered_position_outside_target_falls_back_to_centre(): ...
def test_paused_and_calibrating_never_warp_but_calibrating_publishes_features(): ...
def test_blink_counts_as_no_gaze(): ...                        # eye_openness 0.05 → gaze_updated(None)
def test_focus_failure_still_warps(): ...                      # focus_raises=True → warp recorded, no exception
def test_layout_change_reloads_calibration_and_forgets_vanished_display(): ...
def test_single_display_state(): ...                           # one display → state "single_display", no warps
```

- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement `base.py` and `controller.py` per the spec's Controller section.
  - The controller asks `refresh_displays()` when 2 s have passed since the last refresh.
  - It builds a fresh `PointFilter` and decider state on model change.
  - It wraps `focus_frontmost_window_on` in try/except with `log.exception`.
- [ ] **Step 4:** Implement `macos.py` per the spec's Platform section. Then run `uv run pytest -v`. Expected: all pass, the mac smoke test included.
- [ ] **Step 5: Manual check on this Mac.** Run `uv run python -c "from eyetracker.platform.macos import MacPlatform as P; p=P(); print(p.displays()); print(p.input_idle())"` (expected: the 3 displays in points; idle numbers that reset when you type or move the mouse). Then run `... p.focus_frontmost_window_on(p.displays()[1])` from a terminal on another display. Expected: the front window on display 2 gets focus after Accessibility access is granted to the terminal app.
- [ ] **Step 6: Commit** `feat: macOS platform layer and switching controller`.

---

### Task 10: Camera and vision worker

**Files:**
- Create: `src/eyetracker/vision/camera.py`, `src/eyetracker/vision/worker.py`, `tests/test_worker.py`

**Interfaces:**
- Consumes: `FaceMesh`, `FaceResult` (Task 4); `features_from_landmarks`, `GazeFeatures` (Task 3); `Tuning.fps` (Task 7).
- Produces:
  - `CameraInfo(index: int, name: str, builtin: bool)`.
  - `list_cameras() -> list[CameraInfo]`: AVFoundation video and muxed devices, in OpenCV's index order (verified manually in Step 5).
  - `choose_camera(cams, preferred_name: str | None) -> CameraInfo | None`.
  - `camera_permission() -> str` (`authorized|denied|restricted|not_determined`) and `request_camera_permission(timeout: float = 60) -> bool`.
  - `OpenCVCamera(index, width=640, height=480)` with `open() -> bool`, `read() -> np.ndarray | None`, `release()`.
  - `Observation(t: float, features: GazeFeatures | None, landmarks: np.ndarray | None = None, frame: np.ndarray | None = None)`.
  - `VisionLoop(camera_factory: Callable[[], Camera], face_mesh, clock=time.monotonic)` with `step() -> Observation | None`, attribute `preview: bool`, attribute `status: str` (`"running"` or `"camera_unavailable"`), and `close()`.
  - `VisionWorker(QThread)` wraps a `VisionLoop`: signals `observed(object)` and `status_changed(str)`, methods `stop()` and `set_camera_factory(factory)`. It caps the rate at `Tuning.fps`, and retries a failed open every 5 s with an interruptible wait.

- [ ] **Step 1: Write failing tests:**

```python
def test_choose_camera_prefers_builtin_over_iphone():
    cams = [CameraInfo(0, "iPhone Camera", False), CameraInfo(1, "MacBook Air Camera", True)]
    assert choose_camera(cams, None).index == 1
    assert choose_camera(cams, "iPhone Camera").index == 0
    assert choose_camera(cams, "gone") .index == 1 and choose_camera([], None) is None
def test_step_emits_features_when_face_found(): ...      # fake mesh returns synth_landmarks → obs.features.yaw ≈ 0
def test_step_emits_none_features_without_face(): ...
def test_frame_attached_only_when_preview(): ...
def test_step_reports_unavailable_when_camera_will_not_open():
    loop = VisionLoop(lambda: FakeCamera(opens=False), FakeMesh()); assert loop.step() is None and loop.status == "camera_unavailable"
def test_read_failure_reopens_camera(): ...              # read() None once → factory called again on next step
```

- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. `OpenCVCamera` uses `cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)` and sets width and height. **Step 4:** Run, expect PASS.
- [ ] **Step 5: Manual check.** Run `uv run python -c "from eyetracker.vision.camera import *; print(list_cameras())"`, then open each index with OpenCV and compare against the listed names: the built-in camera shows your face from above the laptop screen. Expected: the order matches. If not, change `list_cameras` ordering and record the rule in a comment.
- [ ] **Step 6: Commit** `feat: camera selection and vision worker thread`.

---

### Task 11: Calibration window

**Files:**
- Create: `src/eyetracker/ui/__init__.py`, `src/eyetracker/ui/calibration_window.py`, `tests/test_calibration_window.py`

**Interfaces:**
- Consumes: `CalibrationSession`, `calibration_points`, `fit_session` (Task 7); `Controller.features_observed`, `set_calibrating` (Task 9).
- Produces: `CalibrationWindow(QObject)`:
  - Constructor `(displays: list[Display], clock=time.monotonic)`.
  - Methods: `begin()` (shows the intro on every display), `start()` (what Space triggers), `cancel()` (what Esc triggers), slot `on_features(f, t)`, and slot `tick()` (driven by the timer; tests call it directly with the fake clock advanced).
  - Signals: `finished(object, object)` (model, report), `failed(str)`, `cancelled()`.
  - Internals: one `_DotWindow(QWidget)` per display (frameless, `WindowStaysOnTopHint`, `setGeometry` to the display rect, never `showFullScreen`), and a 30 ms `QTimer` calling `session.tick`.

- [ ] **Step 1: Write failing tests** (offscreen Qt, fake clock):

```python
def test_full_run_emits_finished(qapp):   # feed 12 features per point from a linear synthetic mapping → finished emitted with report.display_accuracy > 0.9
def test_escape_cancels(qapp): ...        # cancel() → cancelled emitted, all windows hidden
def test_no_face_emits_failed(qapp): ...  # no features → failed("Couldn't see your face...") after all points skipped
```

- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement.
  - Draw a dot of radius 14 pt that shrinks to 6 pt during settle.
  - Use the intro, result and failure copy from the spec. Show the result for 2.5 s, then close.
  - Install a key handler on every window for Space and Esc.
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5: Commit** `feat: per-display calibration window`.

---

### Task 12: Tray, debug views and app wiring

**Files:**
- Create: `src/eyetracker/ui/icons.py`, `src/eyetracker/ui/tray.py`, `src/eyetracker/ui/preview.py`, `src/eyetracker/ui/gaze_dot.py`, `src/eyetracker/app.py`, `src/eyetracker/__main__.py`, `tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `icons.eye_icon(active: bool) -> QIcon` (template/mask image).
  - `Tray(QObject)`:
    - Signals: `pause_toggled(bool)`, `calibrate_requested()`, `camera_selected(str)`, `preview_toggled(bool)`, `gaze_dot_toggled(bool)`, `accessibility_requested()`, `quit_requested()`.
    - Methods: `set_state(state: str)` (maps controller states plus `"camera_unavailable"` to the spec's status text), `set_cameras(names, current)`, `set_accessibility_missing(bool)`, `notify(title, text)`.
  - `PreviewWindow(QWidget)` with `show_observation(obs)`; `GazeDot(QWidget)` with `show_point(point | None)`.
  - `app.main(argv: list[str] | None = None) -> int` with flag `--debug`.

- [ ] **Step 1: Write failing tests:**

```python
def test_tray_signals_and_states(qapp): ...   # clicking Pause action emits pause_toggled(True); set_state("not_calibrated") shows "Not calibrated"
def test_gaze_dot_moves_and_hides(qapp): ...  # show_point((500, 300)) → geometry centred there; None → hidden
def test_preview_renders_observation(qapp): ...
```

- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement the widgets, then `app.main`:
  1. Configure logging.
  2. Create the QApplication with `setQuitOnLastWindowClosed(False)`.
  3. Hide the dock icon.
  4. If models are missing, print `Run: uv run python scripts/fetch_models.py` and exit 1.
  5. Load settings. Request camera permission; if refused, set the tray to "Camera unavailable" and offer to open Camera settings.
  6. Choose the camera and build the FaceMesh, `VisionWorker`, `CalibrationStore`, `Controller` and `Tray`. Connect the signals.
  7. Pause stops the worker, which turns the camera light off. Resume restarts it.
  8. Calibrate: with fewer than 2 displays, notify "Only one display — nothing to switch" and stop. Otherwise `controller.set_calibrating(True)`, then show the `CalibrationWindow`. On finish, `store.save` and `controller.set_model`, then notify with the accuracy. Finish, failure and cancel all call `set_calibrating(False)`.
  9. When uncalibrated with ≥2 displays at start, notify "Choose Calibrate… from the eye icon to set up".
  10. Poll `accessibility_trusted()` every 5 s to update the tray item.
  11. `aboutToQuit` stops the worker.

  `__main__.py` calls `sys.exit(main())`.
- [ ] **Step 4:** Run `uv run pytest -v` (expected: all pass) and `uv run ruff check .` (expected: clean).
- [ ] **Step 5: Commit** `feat: menu bar app, debug views and wiring`.

---

### Task 13: README and end-to-end check on the owner's desk

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`:**
  - What it does, with a short GIF placeholder line.
  - Requirements, then setup: `uv sync`, `uv run python scripts/fetch_models.py`, `uv run eyetracker`.
  - Permissions: Camera and Accessibility are granted to the terminal app when run from source.
  - How to calibrate, and the menu items.
  - Privacy promises (the spec's anti-goals).
  - Roadmap (the spec's non-goals).
  - Credits: MediaPipe Face Landmarker (Apache-2.0), YuNet (MIT).
  - License.
- [ ] **Step 2: End-to-end run with the owner.** On the MacBook Air plus the PL2793H and LG ultrawide, run `uv run eyetracker --debug`. Expected:
  - (a) The eye icon appears in the menu bar and no Dock icon.
  - (b) Camera preview shows the built-in camera with landmarks on the face.
  - (c) Calibrate completes and reports display accuracy ≥ 90 %.
  - (d) With the gaze dot on, the dot lands on the monitor being looked at.
  - (e) Looking at another monitor for about 0.3 s moves the cursor and focus there, and typing goes into that window.
  - (f) Typing while glancing away never moves focus.
  - (g) Pause turns the camera light off.
  - (h) Unplugging a monitor doesn't crash the app.

  Record any failure as a follow-up task, not a silent fix.
- [ ] **Step 3: Commit** `docs: README with setup, permissions and credits`.
