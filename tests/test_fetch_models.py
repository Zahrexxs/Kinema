import io

import pytest

from kinema import fetch_models


def test_model_url_points_at_the_mediapipe_cdn():
    url = fetch_models.model_url("heavy")

    assert url == (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )


def test_download_writes_the_file_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_models.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(b"weights"),
    )
    destination = tmp_path / "pose_landmarker_lite.task"

    fetch_models.download("https://example.test/model", destination)

    assert destination.read_bytes() == b"weights"
    assert not destination.with_suffix(".part").exists()


def test_download_leaves_nothing_behind_on_failure(tmp_path, monkeypatch):
    def fail(url, timeout):
        raise OSError("network down")

    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", fail)
    destination = tmp_path / "pose_landmarker_lite.task"

    with pytest.raises(OSError, match="network down"):
        fetch_models.download("https://example.test/model", destination)

    assert not destination.exists()
    assert not destination.with_suffix(".part").exists()


def test_fetch_skips_variants_already_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_models, "MODELS_DIR", tmp_path)
    (tmp_path / "pose_landmarker_lite.task").write_bytes(b"stub")

    assert fetch_models.fetch(["lite"]) == []
