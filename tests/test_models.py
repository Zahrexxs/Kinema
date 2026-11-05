from pathlib import Path

import pytest

from ahma.core.models import model_filename, resolve_model_path


@pytest.mark.parametrize(
    ("complexity", "expected"),
    [
        (0, "pose_landmarker_lite.task"),
        (1, "pose_landmarker_full.task"),
        (2, "pose_landmarker_heavy.task"),
    ],
)
def test_model_filename_maps_complexity(complexity, expected):
    assert model_filename(complexity) == expected


@pytest.mark.parametrize("complexity", [-1, 3, 99])
def test_invalid_complexity_raises(complexity):
    with pytest.raises(ValueError, match="must be 0, 1 or 2"):
        model_filename(complexity)


def test_resolve_model_path_uses_models_directory():
    path = resolve_model_path("/tmp/ahma/core", 1)

    assert path == Path("/tmp/ahma/core/models/pose_landmarker_full.task")
