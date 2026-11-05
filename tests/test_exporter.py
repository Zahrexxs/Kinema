import csv
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from kinema.core.exporter import MotionExporter


def test_export_csv_writes_one_row_per_frame(tmp_path):
    exporter = MotionExporter()
    exporter.add_frame_data(0, 0.0, {"left_elbow_angle": 90.0})
    exporter.add_frame_data(1, 1 / 30, {"left_elbow_angle": 91.5})

    output = tmp_path / "metrics.csv"
    returned = exporter.export_csv(str(output))

    assert returned == str(output)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["frame"] == "0"
    assert float(rows[1]["timestamp"]) == pytest.approx(1 / 30)
    assert float(rows[1]["left_elbow_angle"]) == pytest.approx(91.5)


def test_generate_report_without_data_returns_empty_string(tmp_path):
    exporter = MotionExporter()

    assert exporter.generate_report("job", str(tmp_path), 1.0) == ""


def test_generate_report_creates_pdf(tmp_path):
    exporter = MotionExporter()
    for frame in range(10):
        exporter.add_frame_data(frame, frame / 30, {"left_elbow_angle": 90 + frame})

    pdf_path = exporter.generate_report("job-1", str(tmp_path), 0.33)

    assert pdf_path == str(tmp_path / "job-1_report.pdf")
    assert Path(pdf_path).is_file()
    assert Path(pdf_path).stat().st_size > 0


def test_tracked_landmarks_are_exported_to_csv(tmp_path):
    exporter = MotionExporter()
    exporter.add_frame_data(0, 0.0, {"left_elbow_angle": 90.0}, tracked=21)

    output = tmp_path / "metrics.csv"
    exporter.export_csv(str(output))

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["tracked_landmarks"] == "21"
    assert rows[0]["person_id"] == "0"


def test_person_id_is_exported(tmp_path):
    exporter = MotionExporter()
    exporter.add_frame_data(0, 0.0, {"left_elbow_angle": 90.0}, tracked=21, person_id=2)

    output = tmp_path / "metrics.csv"
    exporter.export_csv(str(output))

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["person_id"] == "2"


def test_metric_columns_skip_bookkeeping_and_empty_series():
    frame = pd.DataFrame(
        [
            {
                "frame": 0,
                "timestamp": 0.0,
                "person_id": 1,
                "tracked_landmarks": 20,
                "left_elbow_angle": 90.0,
            },
            {
                "frame": 1,
                "timestamp": 1 / 30,
                "person_id": 1,
                "tracked_landmarks": 18,
                "right_knee_angle": None,
            },
        ]
    )

    assert MotionExporter._metric_columns(frame) == ["left_elbow_angle"]


def test_generate_report_with_two_people(tmp_path):
    exporter = MotionExporter()
    for frame in range(4):
        exporter.add_frame_data(
            frame, frame / 30, {"left_knee_angle": 90.0 + frame}, tracked=20, person_id=1
        )
        exporter.add_frame_data(
            frame, frame / 30, {"left_knee_angle": 120.0 + frame}, tracked=21, person_id=2
        )

    pdf_path = exporter.generate_report("two-people", str(tmp_path), 0.2)

    assert Path(pdf_path).is_file()
    assert Path(pdf_path).stat().st_size > 0
    assert not list(tmp_path.glob("*.png"))


def test_generate_report_handles_sparse_metrics_and_cleans_up(tmp_path):
    exporter = MotionExporter()
    exporter.add_frame_data(0, 0.0, {"left_elbow_angle": 90.0}, tracked=20)
    exporter.add_frame_data(1, 1 / 30, {"right_knee_angle": 170.0}, tracked=18)

    pdf_path = exporter.generate_report("job-sparse", str(tmp_path), 0.1)

    assert Path(pdf_path).is_file()
    assert not list(tmp_path.glob("*.png"))


def test_table_widths_fit_the_page():
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)
    rows = [
        ("Metric", "Min", "Max", "Mean"),
        ("right shoulder angle 2d", "10.0", "170.0", "90.0"),
    ]

    widths = MotionExporter._table_widths(pdf, rows)

    assert sum(widths) <= 190.0
    assert all(width > 0 for width in widths)
