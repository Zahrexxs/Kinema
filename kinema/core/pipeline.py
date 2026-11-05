import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from kinema.core.exporter import MotionExporter
from kinema.core.metrics import DEFAULT_MIN_VISIBILITY, MotionAnalyzer
from kinema.core.pose_engine import MediaPipeEstimator, PoseEstimator
from kinema.core.tracking import PersonTracker
from kinema.vis.visualizer import PERSON_COLORS, Visualizer

logger = logging.getLogger(__name__)

FALLBACK_FPS = 30.0
WRITER_CANDIDATES = (("vp80", ".webm"), ("avc1", ".mp4"), ("mp4v", ".mp4"))


@dataclass(frozen=True)
class AnalysisResult:
    video_path: Path
    csv_path: Path | None
    pdf_path: Path | None
    frames: int
    duration: float
    video_duration: float


def _open_writer(
    output_dir: Path, stem: str, fps: float, size: tuple[int, int]
) -> tuple[cv2.VideoWriter, Path]:
    """Open the first video writer whose codec the local OpenCV build supports."""
    width, height = size
    for fourcc, suffix in WRITER_CANDIDATES:
        path = output_dir / f"{stem}_analyzed{suffix}"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc),  # type: ignore[attr-defined]
            fps,
            size,
        )
        if writer.isOpened():
            logger.debug("Writing annotated video with %s to %s", fourcc, path)
            return writer, path
        writer.release()
        path.unlink(missing_ok=True)

    tried = ", ".join(fourcc for fourcc, _ in WRITER_CANDIDATES)
    raise RuntimeError(
        f"No usable video codec ({tried}) for {width}x{height}; "
        "the local OpenCV build may lack ffmpeg support."
    )


def analyze_video(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    output_stem: str | None = None,
    model_complexity: int = 1,
    estimator: PoseEstimator | None = None,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    max_people: int = 1,
    write_csv: bool = True,
    write_pdf: bool = True,
    progress: Callable[[int], None] | None = None,
) -> AnalysisResult:
    """Run the full pose-estimation pipeline over a video file.

    Writes ``{stem}_analyzed.webm|mp4``, ``{stem}_metrics.csv`` and
    ``{stem}_report.pdf`` into ``output_dir``. All resources are released even
    when analysis fails, and a partially written video is removed.

    ``max_people`` above 1 detects and tracks several subjects, giving each a
    stable id in the CSV, its own smoothing and its own set of plots.
    """
    if not 1 <= max_people <= len(PERSON_COLORS):
        raise ValueError(
            f"max_people must be between 1 and {len(PERSON_COLORS)} (got {max_people!r})"
        )
    video_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    owns_estimator = estimator is None
    writer: cv2.VideoWriter | None = None
    annotated_path: Path | None = None
    failure: BaseException | None = None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = FALLBACK_FPS

        if estimator is None:
            estimator = MediaPipeEstimator(
                static_image_mode=False,
                model_complexity=model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                num_poses=max_people,
            )

        tracker = PersonTracker(max_people=max_people)
        analyzers: dict[int, MotionAnalyzer] = {}
        visualizer = Visualizer(min_visibility=min_visibility)
        exporter = MotionExporter()

        writer, annotated_path = _open_writer(out_dir, stem, fps, (width, height))

        started = time.perf_counter()
        frame_count = 0
        detected_frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_s = frame_count / fps
            poses = estimator.process_frame(frame, int(timestamp_s * 1000))
            output_frame = frame.copy()

            if poses:
                detected_frames += 1
                people = tracker.assign(poses)

                for index, person in enumerate(people):
                    analyzer = analyzers.get(person.id)
                    if analyzer is None:
                        analyzer = MotionAnalyzer(fps=fps, smooth_landmarks=True)
                        analyzers[person.id] = analyzer

                    smoothed = analyzer.update_landmarks(person.landmarks, timestamp_s)
                    metrics = analyzer.compute_metrics(smoothed, min_visibility)
                    tracked = analyzer.count_tracked(smoothed, min_visibility)
                    exporter.add_frame_data(
                        frame_count,
                        timestamp_s,
                        metrics,
                        tracked=tracked,
                        person_id=person.id,
                    )

                    output_frame = visualizer.draw_skeleton(
                        output_frame,
                        smoothed,
                        color=PERSON_COLORS[person.id % len(PERSON_COLORS)],
                        label=f"P{person.id}" if len(people) > 1 else None,
                    )
                    # One dashboard only: two would overlap in the same corner.
                    output_frame = visualizer.draw_metrics(
                        output_frame,
                        metrics,
                        smoothed,
                        show_dashboard=(index == 0),
                    )

            writer.write(output_frame)
            frame_count += 1
            if progress is not None:
                progress(frame_count)

        # A clip the model cannot see a person in would otherwise "succeed" with an
        # empty CSV and no report, which reads like a valid measurement.
        if frame_count == 0:
            raise RuntimeError(f"No frames could be read from {video_path}")
        if detected_frames == 0:
            raise RuntimeError(
                f"No pose detected in any of {frame_count} frames of {video_path}. "
                "Check that one person is fully visible for most of the clip."
            )

        duration = time.perf_counter() - started
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if owns_estimator and estimator is not None:
            estimator.close()
        if failure is not None and annotated_path is not None:
            annotated_path.unlink(missing_ok=True)

    csv_path = None
    if write_csv:
        csv_path = Path(exporter.export_csv(str(out_dir / f"{stem}_metrics.csv")))

    pdf_path = None
    if write_pdf:
        report = exporter.generate_report(stem, str(out_dir), duration)
        pdf_path = Path(report) if report else None

    logger.debug("Analyzed %s: %d frames in %.2fs", video_path, frame_count, duration)

    return AnalysisResult(
        video_path=annotated_path,
        csv_path=csv_path,
        pdf_path=pdf_path,
        frames=frame_count,
        duration=duration,
        video_duration=frame_count / fps if fps else 0.0,
    )
