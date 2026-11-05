import argparse

import pytest

from kinema import run
from kinema.core.pipeline import AnalysisResult


def test_output_must_be_a_file(tmp_path):
    with pytest.raises(argparse.ArgumentTypeError, match="must be a file path"):
        run._video_path(str(tmp_path))


def test_output_accepts_a_file_path(tmp_path):
    path = tmp_path / "annotated.mp4"

    assert run._video_path(str(path)) == path


def test_process_video_writes_beside_the_output(synthetic_video, tmp_path, monkeypatch, capsys):
    calls = {}

    def fake_analyze(input_path, output_dir, *, output_stem, model_complexity=1, **kwargs):
        calls.update(
            input_path=input_path,
            output_dir=output_dir,
            output_stem=output_stem,
            model_complexity=model_complexity,
        )
        return AnalysisResult(
            video_path=output_dir / f"{output_stem}_analyzed.webm",
            csv_path=output_dir / f"{output_stem}_metrics.csv",
            pdf_path=output_dir / f"{output_stem}_report.pdf",
            frames=3,
            duration=0.2,
            video_duration=0.1,
        )

    monkeypatch.setattr(run, "analyze_video", fake_analyze)

    result = run.process_video(str(synthetic_video()), str(tmp_path / "out" / "clip.mp4"))

    assert calls["output_dir"] == tmp_path / "out"
    assert calls["output_stem"] == "clip"
    assert result.frames == 3
    printed = capsys.readouterr().out
    assert "clip_analyzed.webm" in printed
    assert "clip_metrics.csv" in printed
    assert "clip_report.pdf" in printed


def test_main_passes_the_resolved_options(monkeypatch, tmp_path):
    calls = {}

    def fake_process_video(input_path, output_path, model_complexity, min_visibility, max_people):
        calls.update(
            input_path=input_path,
            output_path=output_path,
            model_complexity=model_complexity,
            min_visibility=min_visibility,
            max_people=max_people,
        )

    monkeypatch.setattr(run, "process_video", fake_process_video)
    monkeypatch.setattr(
        "sys.argv",
        [
            "kinema",
            "--input",
            "clip.mp4",
            "--output",
            "out.mp4",
            "--complexity",
            "heavy",
            "--max-people",
            "2",
        ],
    )

    run.main()

    assert calls["model_complexity"] == 2
    assert calls["max_people"] == 2
    assert calls["output_path"].endswith("out.mp4")


def test_main_reports_pipeline_errors_without_a_traceback(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Cannot open video: missing.mp4")

    monkeypatch.setattr(run, "process_video", fail)
    monkeypatch.setattr("sys.argv", ["kinema", "--input", "missing.mp4"])

    with pytest.raises(SystemExit, match="Cannot open video"):
        run.main()
