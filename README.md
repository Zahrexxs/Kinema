<div align="center">

# Kinema

**Automated human motion analysis — raw video in, joint angles and PDF reports out.**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
[![CI](https://github.com/Zahrexxs/Kinema/actions/workflows/ci.yml/badge.svg)](https://github.com/Zahrexxs/Kinema/actions/workflows/ci.yml)
![FastAPI](https://img.shields.io/badge/fastapi-0.100%2B-009688?logo=fastapi&logoColor=white)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10%2B-00897B?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

</div>

---

Turns raw video into biomechanical measurements you can actually read: joint-angle
time series, CSV exports, and PDF reports, from a pose-estimation model rather than
a marker or sensor rig.

## Contents

- [Demo](#demo)
- [Why this exists](#why-this-exists)
- [How it works](#how-it-works)
- [Layout](#layout)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Testing and quality](#testing-and-quality)
- [Dependencies](#dependencies)
- [Known limitations](#known-limitations)
- [License](#license)

## Demo

A seven-second barbell back squat, processed with `kinema --complexity full`:

![Annotated frame: skeleton, joint angles and the metrics dashboard](docs/annotated-frame.png)

![Joint angles over time for both knees and both hips](docs/angle-series.png)

The right knee travels from 170° standing to 81° at the bottom of the rep. Gaps in the
curves are frames where a leg was occluded: the angle is dropped rather than guessed,
and `tracked_landmarks` in the CSV records how much of the skeleton was visible.

Demo footage: [Squat – exercise demonstration video](https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm)
by FitnessScape, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), via
Wikimedia Commons.

## Why this exists

Assessing movement from video usually means either a motion-capture rig or a stack of
reflective markers and cameras. Both are expensive and awkward outside a lab. A
monocular pose model gives you a usable approximation from an ordinary video file —
which makes "upload a clip, get numbers back" practical for a physiotherapist, a coach,
or a movement screening session.

The interesting part isn't the pose model. It's everything between *landmarks exist*
and *a person can act on them*: smoothing jittery signals, deriving anatomically
meaningful angles, knowing when a measurement is not trustworthy, and presenting the
result without a data-analysis background.

## How it works

```
video ──▶ OpenCV frame reader
          │
          ▼
    MediaPipeEstimator  ──▶ 33 3D landmarks per frame
          │
          ▼
    OneEuroFilter       ──▶ per-joint smoothing
          │
          ▼
    MotionAnalyzer      ──▶ confidence-gated angles (3D + image plane)
          │
          ├──▶ CSV  (per-frame metrics + tracked-landmark count)
          ├──▶ PNG  (one plot per metric, embedded in the PDF)
          └──▶ PDF  (formatted report)
```

### The smoothing problem

Raw landmark output is noisy. A wrist that is physically still still jitters a few
pixels frame to frame, which shows up as an angle signal that oscillates wildly and
makes every downstream metric useless.

The standard fix is a fixed low-pass filter, but that has a bad trade-off: smoothing
hard enough to kill jitter at rest also smears genuine fast movement.

[`OneEuroFilter`](kinema/core/metrics.py) resolves this by adapting its cutoff to
observed speed — heavy smoothing when a joint is nearly still, light smoothing when it
moves fast. That's what lets the angle series stay flat during a held position without
turning a swing into a ramp.

### The confidence problem

A pose model always returns 33 landmarks, even for body parts it can barely see, and a
low-confidence joint still produces a perfectly well-formed angle. Kinema drops an
angle when any of its three landmarks falls below
[`KINEMA_VISIBILITY_THRESHOLD`](#configuration) (0.5 by default), and records how many
landmarks were trusted for each frame. A blank cell in the CSV means "not tracked",
not "zero degrees".

Every angle is emitted twice: the 3D value, and its projection onto the image plane
(`*_angle_2d`). Monocular depth is inferred, so the 2D column is usually the more
trustworthy one.

## Layout

| Path | What lives there |
|---|---|
| `kinema/core/pipeline.py` | `analyze_video()` — the one frame loop shared by the CLI and the API |
| `kinema/core/pose_engine.py` | `PoseEstimator` ABC + `MediaPipeEstimator` (MediaPipe Tasks API) |
| `kinema/core/metrics.py` | `OneEuroFilter`, `MotionAnalyzer` — smoothing, gating, angles |
| `kinema/core/tracking.py` | `PersonTracker` — stable ids for multi-person clips |
| `kinema/core/exporter.py` | CSV / PNG / PDF output |
| `kinema/core/model_registry.py` | Complexity level → MediaPipe model filename |
| `kinema/config.py` | `KINEMA_*` settings (paths, limits, thresholds, log level) |
| `kinema/api/main.py` | FastAPI service — upload, background analysis, job status |
| `kinema/api/schemas.py` | `Job` state and response models |
| `kinema/vis/visualizer.py` | Annotated-frame rendering |
| `kinema/run.py` | CLI entry point |
| `kinema/fetch_models.py` | Downloads the Pose Landmarker weights |

The `PoseEstimator` interface is deliberate: the analysis pipeline depends on the
abstraction, not on MediaPipe. Swapping in a different pose backend means implementing
one class, not touching `metrics.py`.

## Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Zahrexxs/Kinema.git
cd Kinema

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .

python -m kinema.fetch_models    # lite, full and heavy (~30 MB each)
```

The model weights are gitignored. `fetch_models` pulls them from Google's MediaPipe
CDN and skips variants that already exist. Pass `--force` to re-download, or name the
variants you want (`python -m kinema.fetch_models lite`).

`model_complexity` selects the file: `0` lite, `1` full, `2` heavy. The CLI and the API
both expose it as `lite` / `full` / `heavy`.

## Usage

CLI, over a local file:

```bash
kinema --input path/to/video.mp4 --output out/annotated.mp4
kinema --input path/to/video.mp4 --output out/annotated.mp4 --complexity heavy
kinema --input path/to/video.mp4 --output out/annotated.mp4 --max-people 2
```

`--output` is a **file** path, not a directory. The CSV and PDF are written beside it
using the same stem (`out/annotated_metrics.csv`, `out/annotated_report.pdf`). The
console script is equivalent to `python -m kinema.run`.

HTTP service:

```bash
uvicorn kinema.api.main:app --reload
# then open http://127.0.0.1:8000 and upload a clip
```

| Endpoint | Purpose |
|---|---|
| `GET /` | Uploader UI |
| `GET /health` | Liveness probe for containers and orchestrators |
| `POST /analyze/video` | Multipart `file`, optional `complexity` and `max_people`; returns a job id |
| `GET /status/{job_id}` | `queued` → `processing` → `completed` or `failed` |
| `GET /outputs/{job_id}/{filename}` | Annotated video, CSV or PDF |

Uploads are validated by extension and content type (`415`), capped at
`KINEMA_MAX_UPLOAD_MB` while streaming (`413`), and rejected when empty (`400`). Job
artifacts are kept for `KINEMA_JOB_TTL_HOURS` and swept on the next upload. The job
registry lives in the process, so run a single worker or status polling will miss jobs
handled by another one.

If the model finds no person in any frame, the run fails with
`No pose detected in any of N frames` instead of reporting an empty measurement. An
empty CSV would look like a valid result with zero movement.

### Headless and container deployments

`mediapipe` imports `sounddevice`, which initialises PortAudio at import time even
though Kinema never touches audio, and OpenCV links GL/EGL. A minimal Linux image
needs the same libraries CI installs:

```bash
apt-get install -y --no-install-recommends \
  libgl1 libglib2.0-0 libegl1 libgles2 libportaudio2 pulseaudio
pulseaudio --start --exit-idle-time=-1
```

Without a running PulseAudio server, `import mediapipe` raises `PortAudioError`
before any video is processed.

The repository ships a `Dockerfile` and `docker-compose.yml` that do exactly this:

```bash
docker compose up --build
# http://127.0.0.1:8000
```

The image installs those libraries, bakes in the `lite`/`full`/`heavy` weights
(`--build-arg KINEMA_MODEL_VARIANTS="lite"` for a smaller image), runs as a non-root
user, keeps uploads and reports in the `kinema-data` volume, and exposes the
`/health` probe. CI builds the image and smoke-tests that endpoint, because the
PortAudio failure only shows up in a fresh container.

## Configuration

Every setting is read from the environment (or a `.env` file):

| Variable | Default | Purpose |
|---|---|---|
| `KINEMA_MODEL_COMPLEXITY` | `1` | Default model: `0` lite, `1` full, `2` heavy |
| `KINEMA_VISIBILITY_THRESHOLD` | `0.5` | Minimum landmark confidence for angles and drawing |
| `KINEMA_UPLOAD_DIR` | `<repo>/data/uploads` | Where uploads are stored |
| `KINEMA_OUTPUT_DIR` | `<repo>/data/outputs` | Where artifacts are written |
| `KINEMA_MAX_UPLOAD_MB` | `200` | Upload size limit |
| `KINEMA_JOB_TTL_HOURS` | `24` | Retention for finished jobs and artifacts |
| `KINEMA_CORS_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Allowed origins, comma separated |
| `KINEMA_LOG_LEVEL` | `INFO` | Logging verbosity |

## Outputs

- `*_analyzed.webm` — annotated video. VP8 WebM by default; falls back to H.264 or
  MPEG-4 MP4 when the local OpenCV build cannot encode VP8.
- `*_metrics.csv` — one row per frame **per person**: `frame`, `timestamp`,
  `person_id`, `tracked_landmarks`, six 3D angles and six image-plane angles. Missing
  angles are blank, not zero.
- `*_report.pdf` — summary statistics plus one graph per metric. The PNG plots are
  produced for the report and removed afterwards.

## Testing and quality

```bash
pip install -e ".[dev]"

pytest --cov=kinema --cov-fail-under=85
ruff check . && ruff format --check .
mypy kinema
```

CI runs the suite on Python 3.11 and 3.12, then lint, format, type-check and
`pip-audit` in a separate job. The unit suite needs no model weights and no video
file: a synthetic clip and a fake estimator cover the pipeline end to end.

## Dependencies

Ranges live in [`pyproject.toml`](pyproject.toml); `requirements.txt` and
`requirements-dev.txt` are `pip install -e` shortcuts for them.

[`requirements.lock`](requirements.lock) is a full pinned snapshot for reproducible
deployment, because `mediapipe`, `numpy` and `opencv` break each other's ABI across
releases. It is generated on Python 3.12, so the CI test matrix installs from the
ranges instead — a 3.12 lock pins packages that 3.11 cannot install. The quality job
does install the lock, so it stays exercised. Regenerate it after changing a range:

```bash
pip install -e ".[dev]"
pip freeze --exclude-editable > requirements.lock
```

## Known limitations

Worth stating plainly, since several are real:

- **The 33-landmark skeleton is coarse.** Finger and wrist articulation is not
  resolvable, so "grip strength" or "wrist deviation" style measurements are out of
  scope.
- **2D video means monocular depth is inferred, not measured.** The 3D angle depends on
  an estimated `z` and degrades on off-axis camera angles. Prefer `*_angle_2d` when the
  camera is not square to the movement.
- **Multiple people are opt-in, and identities are approximate.** Pass
  `--max-people 2` (up to 4) and each subject gets its own smoothing, metrics, colour
  and `person_id`. Detections are matched frame to frame by nearest torso centroid, so
  two ids can swap when people cross paths. Use it to measure a session, not to track
  an identity over time.
- **The PDF export is formatting, not analysis.** It reports the numbers; it does not
  interpret them against any clinical or performance baseline.
- **Confidence is MediaPipe's own estimate.** Gating removes obviously bad angles, but a
  joint can be confidently wrong when the subject is occluded in a plausible way.

## License

[MIT](./LICENSE) © 2026 Muhammad Naufal Azzahri
MediaPipe model weights are downloaded separately and carry Google's own license
terms.
