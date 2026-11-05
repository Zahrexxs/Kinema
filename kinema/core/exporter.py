import os
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from fpdf import FPDF
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

NON_METRIC_COLUMNS = {"frame", "timestamp", "person_id", "tracked_landmarks"}
TABLE_WIDTH_MM = 190.0


class MotionExporter:
    def __init__(self) -> None:
        self.data_records: list[dict] = []

    def add_frame_data(
        self,
        frame_idx: int,
        timestamp: float,
        metrics: dict[str, float],
        tracked: int | None = None,
        person_id: int = 0,
    ) -> None:
        """Accumulate data for a single frame and person."""
        record: dict[str, float] = {
            "frame": frame_idx,
            "timestamp": timestamp,
            "person_id": person_id,
        }
        if tracked is not None:
            record["tracked_landmarks"] = tracked
        record.update(metrics)
        self.data_records.append(record)

    def export_csv(self, output_path: str) -> str:
        """Export data to CSV."""
        df = pd.DataFrame(self.data_records)
        df.to_csv(output_path, index=False)
        return output_path

    def _plot_metric(
        self, df: pd.DataFrame, metric: str, path: str, title: str | None = None
    ) -> None:
        figure = Figure(figsize=(10, 4))
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        axes.plot(df["timestamp"], df[metric])
        axes.set_title(title or f"{metric.replace('_', ' ').title()} over Time")
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Angle (deg)")
        axes.grid(True)
        figure.tight_layout()
        figure.savefig(path)

    @staticmethod
    def _metric_columns(df: pd.DataFrame) -> list[str]:
        """Angle columns that actually carry data; bookkeeping and empty ones are skipped."""
        return [
            column
            for column in df.columns
            if column not in NON_METRIC_COLUMNS and df[column].notna().any()
        ]

    @staticmethod
    def _table_widths(pdf: FPDF, rows: Sequence[tuple[str, ...]]) -> list[float]:
        """Fit the table to the page instead of letting long names overflow."""
        widths = []
        for index in range(len(rows[0])):
            longest = max(pdf.get_string_width(row[index]) for row in rows)
            widths.append(longest + 3)

        assigned = sum(widths)
        if assigned > TABLE_WIDTH_MM:
            widths = [width * TABLE_WIDTH_MM / assigned for width in widths]
        return widths

    def generate_report(self, job_id: str, output_dir: str, duration: float) -> str:
        """Generate a PDF report with summary stats and graphs."""
        if not self.data_records:
            return ""

        df = pd.DataFrame(self.data_records)
        report_path = os.path.join(output_dir, f"{job_id}_report.pdf")

        # Frames where a joint was not tracked leave gaps; ignore empty columns.
        metrics = self._metric_columns(df)

        people = sorted(int(person) for person in df["person_id"].unique())
        multiple_people = len(people) > 1

        # 1. Generate Plots
        plot_files = []
        for person in people:
            person_df = df[df["person_id"] == person]
            for metric in metrics:
                if person_df[metric].notna().sum() == 0:
                    continue
                suffix = f"_p{person}" if multiple_people else ""
                plot_path = os.path.join(output_dir, f"{job_id}_{metric}{suffix}.png")
                title = f"{metric.replace('_', ' ').title()} over Time"
                if multiple_people:
                    title = f"Person {person} — {title}"
                self._plot_metric(person_df, metric, plot_path, title=title)
                plot_files.append((metric, person, plot_path))

        # 2. Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Kinema Motion Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Job ID: {job_id}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, f"Duration: {duration:.2f} seconds", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, f"Total Frames: {len(df)}", new_x="LMARGIN", new_y="NEXT")

        if "tracked_landmarks" in df.columns:
            tracked = df["tracked_landmarks"].dropna()
            if not tracked.empty:
                pdf.cell(
                    0,
                    10,
                    f"Mean Tracked Landmarks: {tracked.mean():.0f} / 33",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
        pdf.ln(10)

        # 3. Summary Stats Table
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Summary Statistics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

        rows = [("Metric", "Min", "Max", "Mean")]
        for person in people:
            person_df = df[df["person_id"] == person]
            prefix = f"P{person} " if multiple_people else ""
            for metric in metrics:
                values = person_df[metric].dropna()
                if values.empty:
                    continue
                rows.append(
                    (
                        f"{prefix}{metric.replace('_', ' ')}",
                        f"{values.min():.1f}",
                        f"{values.max():.1f}",
                        f"{values.mean():.1f}",
                    )
                )

        if len(rows) > 1:
            widths = self._table_widths(pdf, rows)
            for row in rows:
                for text, width in zip(row, widths, strict=False):
                    pdf.cell(width, 8, text, border=1)
                pdf.ln(8)

        pdf.ln(10)

        # 4. Embed Plots
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Motion Graphs", new_x="LMARGIN", new_y="NEXT")

        for name, person, path in plot_files:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            heading = name.replace("_", " ").title()
            if multiple_people:
                heading = f"Person {person} — {heading}"
            pdf.cell(
                0,
                10,
                heading,
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.image(path, x=10, w=190)
            # Cleanup temp image
            os.remove(path)

        pdf.output(report_path)
        return report_path
