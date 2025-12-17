import requests

urls = [
    "https://storage.googleapis.com/mediapipe-models/pose/pose_landmarker/float16/1/pose_landmarker_heavy.task",
    "https://storage.googleapis.com/mediapipe-models/pose/pose_landmarker/float16/latest/pose_landmarker_heavy.task",
    "https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_heavy.task",
    "https://storage.googleapis.com/mediapipe-assets/pose_landmarker_heavy.task",
    "https://storage.googleapis.com/mediapipe-models/pose/pose_landmarker/float16/1/pose_landmarker_full.task",
    "https://storage.googleapis.com/mediapipe-models/pose/pose_landmarker/float16/latest/pose_landmarker_full.task"
]

for url in urls:
    try:
        r = requests.head(url, timeout=2)
        print(f"{r.status_code} : {url}")
        if r.status_code == 200:
            print(f"FOUND: {url}")
            break
    except Exception as e:
        print(f"Error {url}: {e}")
