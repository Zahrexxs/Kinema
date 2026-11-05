from pathlib import Path

from kinema.config import Settings


def test_defaults_follow_the_documented_values():
    settings = Settings()

    assert settings.model_complexity == 1
    assert settings.max_upload_mb == 200.0
    assert settings.job_ttl_hours == 24.0
    assert settings.visibility_threshold == 0.5
    assert settings.output_dir == Path(settings.output_dir)
    assert settings.output_dir.parent.name == "data"
    assert settings.upload_dir.parent.name == "data"


def test_environment_overrides_use_the_kinema_prefix(monkeypatch):
    monkeypatch.setenv("KINEMA_MODEL_COMPLEXITY", "2")
    monkeypatch.setenv("KINEMA_MAX_UPLOAD_MB", "15")

    settings = Settings()

    assert settings.model_complexity == 2
    assert settings.max_upload_mb == 15.0


def test_cors_origins_accept_a_comma_separated_value(monkeypatch):
    monkeypatch.setenv("KINEMA_CORS_ORIGINS", "http://a.test, http://b.test")

    settings = Settings()

    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_directories_can_be_pointed_elsewhere(monkeypatch, tmp_path):
    monkeypatch.setenv("KINEMA_UPLOAD_DIR", str(tmp_path / "up"))
    monkeypatch.setenv("KINEMA_OUTPUT_DIR", str(tmp_path / "out"))

    settings = Settings()

    assert settings.upload_dir == tmp_path / "up"
    assert settings.output_dir == tmp_path / "out"
