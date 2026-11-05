import time
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel


class JobState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobState = JobState.QUEUED
    video: str | None = None
    csv: str | None = None
    pdf: str | None = None
    processing_time: float | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class AnalyzeResponse(BaseModel):
    job_id: str
    status: JobState


class JobStatusResponse(BaseModel):
    id: str
    status: JobState
    download_url: str | None = None
    csv_url: str | None = None
    pdf_url: str | None = None
    processing_time: float | None = None
    error: str | None = None
