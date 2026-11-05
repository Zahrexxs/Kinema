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

- [Why this exists](#why-this-exists)
- [How it works](#how-it-works)
- [Layout](#layout)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Outputs](#outputs)
- [Known limitations](#known-limitations)
- [License](#license)

## Why this exists

Assessing movement from video usually means either a motion-capture rig or a stack of
reflective markers and cameras. Both are expensive and awkward outside a lab. A
monocular pose model gives you a usable approximation from an ordinary video file —
which makes "upload a clip, get numbers back" practical for a physiotherapist, a coach,
or a movement screening session.

The interesting part isn't the pose model. It's everything between *landmarks exist*
and *a person can act on them*: smoothing jittery signals, deriving anatomically
meaningful angles, and presenting the result without a data-analysis background.

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
    MotionAnalyzer      ──▶ joint angles (elbow, knee, shoulder…) over time
          │
          ├──▶ CSV  (raw per-frame metrics)
          ├──▶ PNG  (annotated frames / angle plots)
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

## Layout

| Path | What lives there |
|---|---|
| `kinema/core/pose_engine.py` | `PoseEstimator` ABC + `MediaPipeEstimator` (MediaPipe Tasks API) |
| `kinema/core/metrics.py` | `OneEuroFilter`, `MotionAnalyzer` — smoothing and joint-angle derivation |
| `kinema/core/exporter.py` | CSV / PNG / PDF output |
| `kinema/api/main.py` | FastAPI service — upload, background analysis, result retrieval |
| `kinema/vis/visualizer.py` | Annotated-frame rendering |
| `kinema/run.py` | CLI entry point for local batch processing |
| `scripts/fetch_model.sh` | Downloads the Pose Landmarker weights |

The `PoseEstimator` interface is deliberate: the analysis pipeline depends on the
abstraction, not on MediaPipe. Swapping in a different pose backend means implementing
one class, not touching `metrics.py`.

## Setup

```bash
git clone https://github.com/Zahrexxs/Kinema.git
cd Kinema

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/fetch_model.sh
```

The model weights are gitignored; `fetch_model.sh` pulls all three variants from
Google's MediaPipe CDN. `model_complexity` selects the file: `0` lite, `1` full,
`2` heavy.

## Usage

CLI, over a local file:

```bash
python -m kinema.run --input path/to/video.mp4 --output out/
```

HTTP service:

```bash
uvicorn kinema.api.main:app --reload
# then POST a video to /analyze, poll the job, fetch the report
```

## Testing

The unit suite covers the One Euro filter, joint-angle math, model selection and
the exporter. It runs without model weights or a video file.

```bash
pip install -r requirements-dev.txt
pytest
```

CI runs the same command on Python 3.11 for every push and pull request.

## Outputs

- `*_metrics.csv` — per-frame angle values
- `*_angle_plot.png` — angle over time
- `*_report.pdf` — formatted summary
- `*_analyzed.webm` — annotated video (gitignored)

## Known limitations

Worth stating plainly, since several are real:

- **The 33-landmark skeleton is roughly upper- and lower-body coarse.** Finger and
  wrist articulation is not resolvable, so "grip strength" or "wrist deviation" style
  measurements are out of scope.
- **2D video means monocular depth is inferred, not measured.** Joint angles from the
  `z` axis component are estimates and degrade on off-axis camera angles. Angles
  computed in the image plane are far more trustworthy.
- **No multi-person support.** A single `PoseEstimator` tracks one subject; if two
  people are in frame the results are not separable.
- **The PDF export is formatting, not analysis.** It reports the numbers; it does not
  interpret them against any clinical or performance baseline.

## License

[MIT](./LICENSE) © 2026 Muhammad Naufal Azzahri
MediaPipe model weights are downloaded separately and carry Google's own license
terms.
