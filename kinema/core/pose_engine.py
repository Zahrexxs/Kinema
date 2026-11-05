import os
from abc import ABC, abstractmethod

import cv2
import mediapipe as mp
import numpy as np

# Import new Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from kinema.core.model_registry import resolve_model_path
from kinema.core.types import Location


class PoseEstimator(ABC):
    @abstractmethod
    def process_frame(
        self, frame: np.ndarray, timestamp_ms: int | None = None
    ) -> list[dict[str, Location]]:
        """Return one landmark dict per detected person (empty when none)."""
        pass

    @abstractmethod
    def close(self):
        pass

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MediaPipeEstimator(PoseEstimator):
    """MediaPipe Pose implementation using Tasks API (Vision)."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
    ):

        self.static_image_mode = static_image_mode
        self.timestamp_ms = 0  # For video (stream) mode
        self.num_poses = max(1, int(num_poses))

        # Select the model file for the requested complexity (0=lite, 1=full, 2=heavy).
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = str(resolve_model_path(base_dir, model_complexity))

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Run `python -m kinema.fetch_models` to download the model weights."
            )

        # Create options
        base_options = python.BaseOptions(model_asset_path=model_path)

        running_mode = (
            vision.RunningMode.VIDEO if not static_image_mode else vision.RunningMode.IMAGE
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=self.num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,  # proxy for tracking
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # Mapping indices (same as legacy)
        self.landmark_names = {
            0: "nose",
            1: "left_eye_inner",
            2: "left_eye",
            3: "left_eye_outer",
            4: "right_eye_inner",
            5: "right_eye",
            6: "right_eye_outer",
            7: "left_ear",
            8: "right_ear",
            9: "mouth_left",
            10: "mouth_right",
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
            17: "left_pinky",
            18: "right_pinky",
            19: "left_index",
            20: "right_index",
            21: "left_thumb",
            22: "right_thumb",
            23: "left_hip",
            24: "right_hip",
            25: "left_knee",
            26: "right_knee",
            27: "left_ankle",
            28: "right_ankle",
            29: "left_heel",
            30: "right_heel",
            31: "left_foot_index",
            32: "right_foot_index",
        }

    def process_frame(
        self, frame: np.ndarray, timestamp_ms: int | None = None
    ) -> list[dict[str, Location]]:
        # Convert to mp.Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        if self.static_image_mode:
            detection_result = self.landmarker.detect(mp_image)
        else:
            # Video mode requires strictly increasing timestamps. Callers that know
            # the real frame rate pass the true timestamp; otherwise fall back to 30fps.
            if timestamp_ms is None:
                self.timestamp_ms += 33
            else:
                self.timestamp_ms = max(int(timestamp_ms), self.timestamp_ms + 1)
            detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        if not detection_result.pose_landmarks:
            return []

        poses = []
        for landmarks_raw in detection_result.pose_landmarks[: self.num_poses]:
            landmarks = {}
            for idx, lm in enumerate(landmarks_raw):
                name = self.landmark_names.get(idx, f"unknown_{idx}")
                landmarks[name] = Location(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            poses.append(landmarks)

        return poses

    def close(self):
        self.landmarker.close()
