"""Download the MediaPipe Pose Landmarker weights used by MediaPipeEstimator.

The weights are gitignored (the heavy variant alone is ~30 MB), so a fresh
clone needs this step before the CLI or API can analyse a video.
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from kinema.core.model_registry import COMPLEXITY_NAMES

CDN = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
MODELS_DIR = Path(__file__).resolve().parent / "core" / "models"
CHUNK_SIZE = 1 << 20


def model_url(variant: str) -> str:
    return f"{CDN}/pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"


def download(url: str, destination: Path) -> None:
    """Fetch url to destination atomically, so an interrupted run leaves no stub."""
    partial = destination.with_suffix(".part")
    try:
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle, length=CHUNK_SIZE)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def fetch(variants: list[str], force: bool = False) -> list[Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    for variant in variants:
        destination = MODELS_DIR / f"pose_landmarker_{variant}.task"
        if destination.exists() and not force:
            print(f"{variant}: already present, skipping")
            continue

        print(f"{variant}: downloading")
        download(model_url(variant), destination)
        print(f"{variant}: {destination.stat().st_size / 1e6:.1f} MB")
        saved.append(destination)

    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "variants",
        nargs="*",
        choices=list(COMPLEXITY_NAMES),
        help="Variants to download (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download variants that already exist"
    )
    args = parser.parse_args()

    variants = args.variants or list(COMPLEXITY_NAMES)

    try:
        saved = fetch(variants, force=args.force)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Models ready in {MODELS_DIR}")
    if saved:
        print(f"Downloaded {len(saved)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
