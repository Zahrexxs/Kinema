from pathlib import Path

MODEL_FILENAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}


def model_filename(model_complexity: int) -> str:
    """Return the MediaPipe model filename for a complexity level (0-2)."""
    try:
        return MODEL_FILENAMES[model_complexity]
    except KeyError:
        raise ValueError(
            f"model_complexity must be 0, 1 or 2 (got {model_complexity!r})"
        ) from None


def resolve_model_path(base_dir: str | Path, model_complexity: int) -> Path:
    """Resolve the model path for a complexity level relative to a component dir."""
    return Path(base_dir) / "models" / model_filename(model_complexity)
