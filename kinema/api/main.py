import os
import shutil
import uuid
from typing import Dict

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from kinema.core.pipeline import analyze_video

# Initialize App
app = FastAPI(title="Kinema: Automated Human Motion Analysis")

# CORS (Allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount Static Files (Frontend)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# Job Storage (In-memory for MVP)
jobs: Dict[str, str] = {} # job_id -> status

def process_video_task(job_id: str, input_path: str):
    """Background task wrapper around the shared analysis pipeline."""
    try:
        jobs[job_id] = "processing"

        result = analyze_video(
            input_path,
            OUTPUT_DIR,
            output_stem=job_id,
            model_complexity=1, # 1 for speed
        )

        csv_name = result.csv_path.name if result.csv_path else "none"
        pdf_name = result.pdf_path.name if result.pdf_path else "none"

        # Format: completed:filename:duration:csv_name:pdf_name
        jobs[job_id] = (
            f"completed:{result.video_path.name}:{result.duration:.2f}:"
            f"{csv_name}:{pdf_name}"
        )

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        jobs[job_id] = f"failed: {str(e)}"

@app.post("/analyze/video")
async def analyze_video_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    
    # Save Upload
    file_ext = os.path.splitext(file.filename)[1]
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{file_ext}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    jobs[job_id] = "queued"
    
    # Start Task
    background_tasks.add_task(process_video_task, job_id, input_path)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    status = jobs.get(job_id, "unknown")
    if status.startswith("completed:"):
        parts = status.split(":")
        filename = parts[1]
        duration = parts[2] if len(parts) > 2 else "N/A"
        csv_link = f"/outputs/{parts[3]}" if len(parts) > 3 else None
        pdf_link = f"/outputs/{parts[4]}" if len(parts) > 4 and parts[4] != "none" else None
        
        return {
            "id": job_id, 
            "status": "completed", 
            "download_url": f"/outputs/{filename}",
            "processing_time": duration,
            "csv_url": csv_link,
            "pdf_url": pdf_link
        }
    return {"id": job_id, "status": status}

@app.get("/outputs/{filename}")
def get_output(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")
