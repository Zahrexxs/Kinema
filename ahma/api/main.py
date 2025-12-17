import os
import shutil
import uuid
from typing import Dict

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ahma.core.pose_engine import MediaPipeEstimator
from ahma.core.metrics import MotionAnalyzer
from ahma.vis.visualizer import Visualizer

# Initialize App
app = FastAPI(title="AHMA: Automated Human Motion Analysis")

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

def process_video_task(job_id: str, input_path: str, output_path: str):
    """Background task wrapper for video processing."""
    try:
        jobs[job_id] = "processing"
        
        # NOTE: In a real system, we'd import the pipeline logic from run.py or refactor it to a function
        # Since run.py has process_video, let's reuse a similar logic here or import it if modular enough.
        # Ideally, I should have put process_video in a separate lib module.
        # For now, I will re-implement the loop here to ensure clean integration with no stdout spam.
        
        import cv2
        import time
        from ahma.core.pose_engine import MediaPipeEstimator
        from ahma.core.metrics import MotionAnalyzer
        from ahma.vis.visualizer import Visualizer
        from ahma.core.exporter import MotionExporter

        start_time = time.time()


        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            jobs[job_id] = "failed: cannot open video"
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0

        pose_estimator = MediaPipeEstimator(model_complexity=1) # 1 for speed
        motion_analyzer = MotionAnalyzer(fps=fps, smooth_landmarks=True)
        visualizer = Visualizer()
        exporter = MotionExporter()

        fourcc = cv2.VideoWriter_fourcc(*'avc1') # H.264 for web compatibility (if opencv supports it)
        # Fallback to mp4v if avc1 fails in environment
        # Note: Browsers like mp4 (h264). mp4v might not play in all browsers. 
        # VP8/9 (webm) is safer for web if ffmpeg is available.
        
        # Let's try to infer a good codec or just use mp4v and hope browser supports it (Chrome often does)
        # Using 'vp80' => .webm is often safer for immediate web playback without transcoding.
        output_path_webm = output_path.replace(".mp4", ".webm")
        out = cv2.VideoWriter(output_path_webm, cv2.VideoWriter_fourcc(*'vp80'), fps, (width, height))
        
        if not out.isOpened():
             # Fallback to mp4v
             out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
             final_output = output_path
        else:
             final_output = output_path_webm

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = pose_estimator.process_frame(frame)
            output_frame = frame.copy()
            
            if landmarks:
                smoothed_landmarks = motion_analyzer.update_landmarks(landmarks)
                metrics = motion_analyzer.compute_metrics(smoothed_landmarks)
                
                # Accumulate Data
                current_time = frame_count / fps
                exporter.add_frame_data(frame_count, current_time, metrics)
                
                output_frame = visualizer.draw_skeleton(output_frame, smoothed_landmarks)
                output_frame = visualizer.draw_metrics(output_frame, metrics, smoothed_landmarks)
            
            out.write(output_frame)
            frame_count += 1
            
        cap.release()
        out.release()
        pose_estimator.close()
        
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Generate Reports
        csv_path = os.path.join(OUTPUT_DIR, f"{job_id}_metrics.csv")
        exporter.export_csv(csv_path)
        
        # NOTE: PDF generation might differ in path logic
        # For simplicity, exporter uses output_dir + job_id
        
        # We manually call generate_report but it writes to file
        pdf_path = exporter.generate_report(job_id, OUTPUT_DIR, duration)
        
        # Format: completed:filename:duration:csv_name:pdf_name
        csv_name = os.path.basename(csv_path)
        pdf_name = os.path.basename(pdf_path) if pdf_path else "none"
        
        jobs[job_id] = f"completed:{os.path.basename(final_output)}:{duration:.2f}:{csv_name}:{pdf_name}"
        
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        jobs[job_id] = f"failed: {str(e)}"

@app.post("/analyze/video")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    
    # Save Upload
    file_ext = os.path.splitext(file.filename)[1]
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{file_ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_analyzed.mp4")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    jobs[job_id] = "queued"
    
    # Start Task
    background_tasks.add_task(process_video_task, job_id, input_path, output_path)
    
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
