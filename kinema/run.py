import argparse
from pathlib import Path

from kinema.core.pipeline import AnalysisResult, analyze_video


def _video_path(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        raise argparse.ArgumentTypeError(
            f"--output must be a file path, got a directory: {path}"
        )
    return path


def _print_progress(frame_count: int) -> None:
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...", end="\r", flush=True)


def process_video(input_path: str, output_path: str) -> AnalysisResult:
    """Analyze a video, writing the annotated video and reports beside output_path."""
    output = Path(output_path)
    result = analyze_video(
        input_path,
        output.parent,
        output_stem=output.stem,
        progress=_print_progress,
    )
    print()
    print(f"Annotated video: {result.video_path}")
    if result.csv_path is not None:
        print(f"Metrics CSV:     {result.csv_path}")
    if result.pdf_path is not None:
        print(f"PDF report:      {result.pdf_path}")
    print(
        f"Processed {result.frames} frames in {result.duration:.2f}s "
        f"({result.video_duration:.2f}s of video)"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Human Motion Analysis System")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--output",
        type=_video_path,
        default=Path("output.mp4"),
        help="Path for the annotated video; CSV and PDF are written beside it",
    )

    args = parser.parse_args()

    try:
        process_video(args.input, str(args.output))
    except (RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
