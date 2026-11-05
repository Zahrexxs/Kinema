import csv

import pytest
from conftest import FakeEstimator, loc

from kinema.core.pipeline import AnalysisResult, analyze_video


class RecordingEstimator(FakeEstimator):
    def __init__(self, landmarks=None):
        super().__init__(landmarks)
        self.timestamps = []

    def process_frame(self, frame, timestamp_ms=None):
        self.timestamps.append(timestamp_ms)
        return super().process_frame(frame, timestamp_ms)


class ExplodingEstimator(FakeEstimator):
    def process_frame(self, frame, timestamp_ms=None):
        raise RuntimeError("boom")


def test_analyze_video_writes_all_artifacts(synthetic_video, tmp_path):
    video = synthetic_video(frames=6)
    output_dir = tmp_path / "out"

    result = analyze_video(video, output_dir, estimator=FakeEstimator())

    assert isinstance(result, AnalysisResult)
    assert result.frames == 6
    assert result.video_duration == pytest.approx(6 / 30)
    assert result.video_path.parent == output_dir
    assert result.video_path.suffix in {".webm", ".mp4"}
    assert result.video_path.is_file()
    assert result.video_path.stat().st_size > 0

    assert result.csv_path == output_dir / "clip_metrics.csv"
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert "left_elbow_angle" in rows[0]
    assert float(rows[3]["timestamp"]) == pytest.approx(3 / 30)

    assert result.pdf_path == output_dir / "clip_report.pdf"
    assert result.pdf_path.is_file()
    assert result.pdf_path.stat().st_size > 0


def test_analyze_video_can_skip_reports(synthetic_video, tmp_path):
    result = analyze_video(
        synthetic_video(),
        tmp_path / "out",
        estimator=FakeEstimator(),
        write_csv=False,
        write_pdf=False,
    )

    assert result.csv_path is None
    assert result.pdf_path is None
    assert not list(tmp_path.glob("**/*.csv"))


def test_analyze_video_uses_output_stem(synthetic_video, tmp_path):
    result = analyze_video(
        synthetic_video(),
        tmp_path / "out",
        output_stem="job-123",
        estimator=FakeEstimator(),
        write_pdf=False,
    )

    assert result.csv_path.name == "job-123_metrics.csv"
    assert result.video_path.name.startswith("job-123_analyzed")


def test_analyze_video_passes_real_timestamps(synthetic_video, tmp_path):
    estimator = RecordingEstimator()

    analyze_video(
        synthetic_video(frames=6, fps=60.0),
        tmp_path / "out",
        estimator=estimator,
        write_pdf=False,
    )

    assert estimator.timestamps == [0, 16, 33, 50, 66, 83]
    assert estimator.closed is False


def test_analyze_video_raises_for_unreadable_input(tmp_path):
    missing = tmp_path / "missing.mp4"

    with pytest.raises(RuntimeError, match="Cannot open video"):
        analyze_video(missing, tmp_path / "out", estimator=FakeEstimator())


def test_failure_releases_writer_and_removes_partial_file(synthetic_video, tmp_path):
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="boom"):
        analyze_video(synthetic_video(), output_dir, estimator=ExplodingEstimator())

    assert not list(output_dir.glob("*_analyzed.*"))


def test_owned_estimator_is_closed_on_failure(synthetic_video, tmp_path, monkeypatch):
    closed = []

    class Probe(FakeEstimator):
        def process_frame(self, frame, timestamp_ms=None):
            raise RuntimeError("boom")

    def fake_estimator(**kwargs):
        estimator = Probe()
        original_close = estimator.close

        def close():
            closed.append(True)
            original_close()

        estimator.close = close
        return estimator

    monkeypatch.setattr("kinema.core.pipeline.MediaPipeEstimator", fake_estimator)

    with pytest.raises(RuntimeError, match="boom"):
        analyze_video(synthetic_video(), tmp_path / "out")

    assert closed == [True]


def test_metrics_accumulate_only_while_tracked(synthetic_video, tmp_path):
    irrelevant = FakeEstimator(landmarks={"nose": loc(0.5, 0.5)})

    result = analyze_video(
        synthetic_video(frames=4),
        tmp_path / "out",
        estimator=irrelevant,
        write_pdf=False,
    )

    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "left_elbow_angle" not in rows[0]


def test_csv_reports_tracked_landmarks_and_2d_angles(synthetic_video, tmp_path):
    result = analyze_video(
        synthetic_video(frames=3),
        tmp_path / "out",
        estimator=FakeEstimator(),
        write_pdf=False,
    )

    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["tracked_landmarks"] == "3"
    assert "left_elbow_angle" in rows[0]
    assert "left_elbow_angle_2d" in rows[0]


def test_low_visibility_landmarks_produce_no_angles(synthetic_video, tmp_path):
    faint = {
        name: loc(value.x, value.y, value.z, 0.1)
        for name, value in FakeEstimator().landmarks.items()
    }

    result = analyze_video(
        synthetic_video(frames=3),
        tmp_path / "out",
        estimator=FakeEstimator(landmarks=faint),
        write_pdf=False,
    )

    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert "left_elbow_angle" not in rows[0]
    assert rows[0]["tracked_landmarks"] == "0"
