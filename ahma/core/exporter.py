import pandas as pd
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from typing import List, Dict
import time

class MotionExporter:
    def __init__(self):
        self.data_records = []

    def add_frame_data(self, frame_idx: int, timestamp: float, metrics: Dict[str, float]):
        """Accumulate data for a single frame."""
        record = {
            "frame": frame_idx,
            "timestamp": timestamp,
            **metrics
        }
        self.data_records.append(record)

    def export_csv(self, output_path: str) -> str:
        """Export data to CSV."""
        df = pd.DataFrame(self.data_records)
        df.to_csv(output_path, index=False)
        return output_path

    def generate_report(self, job_id: str, output_dir: str, duration: float) -> str:
        """Generate a PDF report with summary stats and graphs."""
        if not self.data_records:
            return ""

        df = pd.DataFrame(self.data_records)
        report_path = os.path.join(output_dir, f"{job_id}_report.pdf")
        
        # 1. Generate Plots
        plot_files = []
        metrics = [c for c in df.columns if c not in ['frame', 'timestamp']]
        
        for metric in metrics:
            plt.figure(figsize=(10, 4))
            plt.plot(df['timestamp'], df[metric])
            plt.title(f"{metric} over Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle (deg)")
            plt.grid(True)
            
            plot_path = os.path.join(output_dir, f"{job_id}_{metric}.png")
            plt.savefig(plot_path)
            plt.close()
            plot_files.append((metric, plot_path))

        # 2. Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"AHMA Motion Analysis Report", ln=True, align='C')
        
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Job ID: {job_id}", ln=True)
        pdf.cell(0, 10, f"Duration: {duration:.2f} seconds", ln=True)
        pdf.cell(0, 10, f"Total Frames: {len(df)}", ln=True)
        pdf.ln(10)

        # 3. Summary Stats Table
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Summary Statistics", ln=True)
        pdf.set_font("Arial", '', 10)
        
        # Header
        col_width = 40
        pdf.cell(col_width, 10, "Metric", border=1)
        pdf.cell(col_width, 10, "Min", border=1)
        pdf.cell(col_width, 10, "Max", border=1)
        pdf.cell(col_width, 10, "Mean", border=1)
        pdf.ln()
        
        for metric in metrics:
            stats = df[metric].describe()
            pdf.cell(col_width, 10, metric.replace('_', ' '), border=1)
            pdf.cell(col_width, 10, f"{stats['min']:.1f}", border=1)
            pdf.cell(col_width, 10, f"{stats['max']:.1f}", border=1)
            pdf.cell(col_width, 10, f"{stats['mean']:.1f}", border=1)
            pdf.ln()
            
        pdf.ln(10)
        
        # 4. Embed Plots
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Motion Graphs", ln=True)
        
        for name, path in plot_files:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, name.replace('_', ' ').title(), ln=True)
            pdf.image(path, x=10, w=190)
            # Cleanup temp image
            os.remove(path)
            
        pdf.output(report_path)
        return report_path
