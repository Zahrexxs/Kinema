import logging
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kinema.api.schemas import AnalyzeResponse, Job, JobState, JobStatusResponse
from kinema.config import get_settings
from kinema.core.model_registry import complexity_from_name
from kinema.core.pipeline import analyze_video

logger = logging.getLogger(__name__)

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

UPLOAD_DIR = Path(settings.upload_dir)
OUTPUT_DIR = Path(settings.output_dir)
STATIC_DIR = Path(__file__).resolve().parent / "static"

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
COPY_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024

app = FastAPI(
    title="Kinema: Automated Human Motion Analysis",
    description="Upload a video, get joint-angle metrics, an annotated video and a PDF report.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_job(job_id: str) -> Job | None:
    with jobs_lock:
        return jobs.get(job_id)


def _update_job(job_id: str, **fields) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        for name, value in fields.items():
            setattr(job, name, value)


def _sweep_expired() -> None:
    """Drop finished jobs and their artifacts once they pass the retention window."""
    cutoff = time.time() - settings.job_ttl_hours * 3600
    with jobs_lock:
        expired = [
            job_id
            for job_id, job in jobs.items()
            if job.status in (JobState.COMPLETED, JobState.FAILED) and job.created_at < cutoff
        ]
        for job_id in expired:
            jobs.pop(job_id, None)

    for job_id in expired:
        for directory in (UPLOAD_DIR, OUTPUT_DIR):
            shutil.rmtree(directory / job_id, ignore_errors=True)


def _validated_suffix(file: UploadFile) -> str:
    requested = Path(file.filename or "").suffix.lower()
    # Pick the canonical value out of the allowlist rather than echoing the
    # request back; only these five strings can influence a filesystem path.
    suffix = next((ext for ext in ALLOWED_EXTENSIONS if ext == requested), None)
    if suffix is None:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=415,
            detail=(f"Unsupported video type '{requested or file.filename}'. Allowed: {allowed}"),
        )

    content_type = file.content_type or ""
    if (
        content_type
        and not content_type.startswith("video/")
        and content_type != "application/octet-stream"
    ):
        raise HTTPException(status_code=415, detail=f"Unsupported content type '{content_type}'")

    return suffix


def _save_upload(file: UploadFile, destination: Path) -> None:
    total = 0
    try:
        with destination.open("wb") as buffer:
            while chunk := file.file.read(COPY_CHUNK_SIZE):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {settings.max_upload_mb:g} MB limit",
                    )
                buffer.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise

    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")


def process_video_task(
    job_id: str, input_path: Path, output_dir: Path, model_complexity: int
) -> None:
    """Background task wrapper around the shared analysis pipeline."""
    _update_job(job_id, status=JobState.PROCESSING)

    try:
        result = analyze_video(
            input_path,
            output_dir,
            output_stem=job_id,
            model_complexity=model_complexity,
            min_visibility=settings.visibility_threshold,
        )
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        _update_job(job_id, status=JobState.FAILED, error=str(exc))
        return

    _update_job(
        job_id,
        status=JobState.COMPLETED,
        video=result.video_path.name,
        csv=result.csv_path.name if result.csv_path else None,
        pdf=result.pdf_path.name if result.pdf_path else None,
        processing_time=round(result.duration, 2),
    )


@app.get("/health")
def health():
    """Liveness probe for containers and orchestrators."""
    return {"status": "ok"}


@app.get("/")
def read_root():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Frontend is not installed")
    return FileResponse(index)


@app.post("/analyze/video", response_model=AnalyzeResponse)
async def analyze_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    complexity: str | None = Form(default=None),
):
    _ensure_dirs()
    _sweep_expired()

    suffix = _validated_suffix(file)

    try:
        model_complexity = (
            settings.model_complexity if complexity is None else complexity_from_name(complexity)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    job_id = str(uuid.uuid4())
    upload_dir = UPLOAD_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / f"input{suffix}"

    try:
        _save_upload(file, input_path)
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    with jobs_lock:
        jobs[job_id] = Job(id=job_id)

    background_tasks.add_task(
        process_video_task, job_id, input_path, OUTPUT_DIR / job_id, model_complexity
    )

    return AnalyzeResponse(job_id=job_id, status=JobState.QUEUED)


@app.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")

    if job.status is not JobState.COMPLETED:
        return JobStatusResponse(id=job.id, status=job.status, error=job.error)

    def artifact_url(name: str | None) -> str | None:
        return f"/outputs/{job.id}/{name}" if name else None

    return JobStatusResponse(
        id=job.id,
        status=job.status,
        download_url=artifact_url(job.video),
        csv_url=artifact_url(job.csv),
        pdf_url=artifact_url(job.pdf),
        processing_time=job.processing_time,
    )


@app.get("/outputs/{job_id}/{filename}")
def get_output(job_id: str, filename: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="File not found")

    # The request selects an artifact the job already produced; it never supplies
    # a path. Anything else is a 404, including names that exist on disk.
    artifact = next((name for name in (job.video, job.csv, job.pdf) if name == filename), None)
    if artifact is None:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = OUTPUT_DIR / job.id / artifact
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)
