import cv2
import argparse
import sys
import time
from ahma.core.pose_engine import MediaPipeEstimator
from ahma.core.metrics import MotionAnalyzer
from ahma.vis.visualizer import Visualizer

def process_video(input_path: str, output_path: str):
    print(f"Processing video: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    # Initialize components
    pose_estimator = MediaPipeEstimator(
        static_image_mode=False,
        model_complexity=1, # 1 for speed during dev, 2 for accuracy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    motion_analyzer = MotionAnalyzer(fps=fps, smooth_landmarks=True)
    visualizer = Visualizer()

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 1. Pose Estimation
            landmarks = pose_estimator.process_frame(frame)
            
            output_frame = frame.copy()
            
            if landmarks:
                # 2. Metric Computation
                # Apply smoothing
                smoothed_landmarks = motion_analyzer.update_landmarks(landmarks)
                
                # Compute angles/stats
                metrics = motion_analyzer.compute_metrics(smoothed_landmarks)
                
                # 3. Visualization
                # Draw skeleton
                output_frame = visualizer.draw_skeleton(output_frame, smoothed_landmarks)
                # Draw metrics
                output_frame = visualizer.draw_metrics(output_frame, metrics, smoothed_landmarks)
            
            out.write(output_frame)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...", end='\r')

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nFinished. Processed {frame_count} frames in {duration:.2f}s ({frame_count/duration:.2f} fps)")
        
        cap.release()
        out.release()
        pose_estimator.close()
        print(f"Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Automated Human Motion Analysis System")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video file")
    
    args = parser.parse_args()
    
    process_video(args.input, args.output)

if __name__ == "__main__":
    main()
