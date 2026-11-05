import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from kinema.api import main as api
from kinema.core.pipeline import AnalysisResult


def stub_pipeline(monkeypatch, error=None):
    calls = []

    def fake_analyze(input_path, output_dir, *, output_stem, model_complexity=1, **kwargs):
        calls.append(
            {
                "input_path": Path(input_path),
                "output_dir": Path(output_dir),
                "model_complexity": model_complexity,
            }
        )
        if error is not None:
            raise error

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        video = output / f"{output_stem}_analyzed.webm"
        video.write_bytes(b"annotated")
        csv = output / f"{output_stem}_metrics.csv"
        csv.write_text("frame,timestamp\n0,0.0\n", encoding="utf-8")
        pdf = output / f"{output_stem}_report.pdf"
        pdf.write_bytes(b"%PDF-1.4 stub")

        return AnalysisResult(
            video_path=video,
            csv_path=csv,
            pdf_path=pdf,
            frames=1,
            duration=0.5,
            video_duration=1 / 30,
        )

    monkeypatch.setattr(api, "analyze_video", fake_analyze)
    return calls


@pytest.fixture
def server(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    jobs = {}
    monkeypatch.setattr(api, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(api, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(api, "jobs", jobs)

    return SimpleNamespace(
        client=TestClient(api.app), uploads=uploads, outputs=outputs, jobs=jobs
    )


def upload(server, name="clip.mp4", content=b"fake video", content_type="video/mp4", **data):
    return server.client.post(
        "/analyze/video", files={"file": (name, content, content_type)}, data=data
    )


def test_upload_completes_and_serves_artifacts(server, monkeypatch):
    calls = stub_pipeline(monkeypatch)

    response = upload(server)

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "queued"

    status = server.client.get(f"/status/{job_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "completed"
    assert body["processing_time"] == 0.5
    assert body["download_url"] == f"/outputs/{job_id}/{job_id}_analyzed.webm"
    assert body["csv_url"] == f"/outputs/{job_id}/{job_id}_metrics.csv"
    assert body["pdf_url"] == f"/outputs/{job_id}/{job_id}_report.pdf"
    assert body["error"] is None

    assert calls[0]["input_path"] == server.uploads / job_id / "input.mp4"
    assert calls[0]["output_dir"] == server.outputs / job_id
    assert calls[0]["model_complexity"] == 1

    assert server.client.get(body["download_url"]).content == b"annotated"
    assert server.client.get(body["csv_url"]).status_code == 200
    assert server.client.get(body["pdf_url"]).status_code == 200


def test_failure_is_reported_without_parsing_errors(server, monkeypatch):
    stub_pipeline(monkeypatch, error=RuntimeError("cannot open: weird: name"))

    job_id = upload(server).json()["job_id"]
    body = server.client.get(f"/status/{job_id}").json()

    assert body["status"] == "failed"
    assert body["error"] == "cannot open: weird: name"
    assert body["download_url"] is None
    assert body["csv_url"] is None


def test_unknown_job_returns_404(server):
    assert server.client.get("/status/does-not-exist").status_code == 404


def test_missing_artifact_returns_404(server):
    assert server.client.get("/outputs/some-job/missing.csv").status_code == 404


@pytest.mark.parametrize(
    ("job_id", "filename"),
    [("..", "passwd"), ("job", "..\\secret"), ("job", "a:b"), ("job/..", "x")],
)
def test_output_rejects_unsafe_names(server, job_id, filename):
    with pytest.raises(HTTPException) as excinfo:
        api.get_output(job_id, filename)

    assert excinfo.value.status_code == 404


def test_unsupported_extension_returns_415(server):
    response = upload(server, name="notes.txt", content=b"hi", content_type="text/plain")

    assert response.status_code == 415
    assert "Unsupported video type" in response.json()["detail"]


def test_non_video_content_type_returns_415(server):
    response = upload(server, name="clip.mp4", content=b"hi", content_type="text/plain")

    assert response.status_code == 415
    assert "content type" in response.json()["detail"]


def test_oversized_upload_returns_413(server, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)

    response = upload(server, content=b"0123456789")

    assert response.status_code == 413
    assert server.jobs == {}
    assert not server.uploads.exists() or not any(server.uploads.iterdir())


def test_empty_upload_returns_400(server):
    response = upload(server, content=b"")

    assert response.status_code == 400
    assert server.jobs == {}


def test_complexity_name_is_forwarded(server, monkeypatch):
    calls = stub_pipeline(monkeypatch)

    upload(server, complexity="heavy")

    assert calls[0]["model_complexity"] == 2


def test_invalid_complexity_returns_400(server, monkeypatch):
    stub_pipeline(monkeypatch)

    response = upload(server, complexity="ultra")

    assert response.status_code == 400
    assert "complexity must be one of" in response.json()["detail"]


def test_expired_jobs_and_artifacts_are_swept(server, monkeypatch):
    stub_pipeline(monkeypatch)
    job_id = upload(server).json()["job_id"]
    assert (server.outputs / job_id).is_dir()

    server.jobs[job_id].created_at = time.time() - 48 * 3600
    api._sweep_expired()

    assert job_id not in server.jobs
    assert not (server.outputs / job_id).exists()
    assert not (server.uploads / job_id).exists()


def test_index_is_served(server):
    response = server.client.get("/")

    assert response.status_code == 200
    assert "Kinema" in response.text
