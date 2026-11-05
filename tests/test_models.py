from pathlib import Path

import pytest

from kinema.core.models import complexity_from_name, model_filename, resolve_model_path


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
    path = resolve_model_path("/tmp/kinema/core", 1)

    assert path == Path("/tmp/kinema/core/models/pose_landmarker_full.task")


@pytest.mark.parametrize(("name", "level"), [("lite", 0), ("full", 1), ("HEAVY", 2)])
def test_complexity_from_name_maps_variants(name, level):
    assert complexity_from_name(name) == level


def test_unknown_complexity_name_raises():
    with pytest.raises(ValueError, match="complexity must be one of"):
        complexity_from_name("ultra")
