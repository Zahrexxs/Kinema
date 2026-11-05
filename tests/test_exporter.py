import csv
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from ahma.core.exporter import MotionExporter


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
