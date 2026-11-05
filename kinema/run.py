import argparse
import logging
from pathlib import Path

from kinema.config import get_settings
from kinema.core.metrics import DEFAULT_MIN_VISIBILITY
from kinema.core.model_registry import COMPLEXITY_NAMES, complexity_from_name
from kinema.core.pipeline import AnalysisResult, analyze_video

logger = logging.getLogger(__name__)


def _video_path(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        raise argparse.ArgumentTypeError(
            f"--output must be a file path, got a directory: {path}"
        )
    return path


def _log_progress(frame_count: int) -> None:
    if frame_count % 30 == 0:
        logger.info("Processed %d frames", frame_count)


def process_video(
    input_path: str,
    output_path: str,
    model_complexity: int = 1,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
) -> AnalysisResult:
    """Analyze a video, writing the annotated video and reports beside output_path."""
    output = Path(output_path)
    result = analyze_video(
        input_path,
        output.parent,
        output_stem=output.stem,
        model_complexity=model_complexity,
        min_visibility=min_visibility,
        progress=_log_progress,
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
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Automated Human Motion Analysis System")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--output",
        type=_video_path,
        default=Path("output.mp4"),
        help="Path for the annotated video; CSV and PDF are written beside it",
    )
    parser.add_argument(
        "--complexity",
        choices=list(COMPLEXITY_NAMES),
        default=None,
        help="Pose model size (default: KINEMA_MODEL_COMPLEXITY, else full)",
    )

    args = parser.parse_args()

    if args.complexity is None:
        model_complexity = settings.model_complexity
    else:
        model_complexity = complexity_from_name(args.complexity)

    try:
        process_video(
            args.input,
            str(args.output),
            model_complexity,
            settings.visibility_threshold,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
