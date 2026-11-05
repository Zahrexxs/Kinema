import cv2
import numpy as np
import pytest

from kinema.core.pose_engine import PoseEstimator
from kinema.core.types import Location


def loc(x, y, z=0.0, visibility=1.0):
    return Location(x, y, z, visibility)


class FakeEstimator(PoseEstimator):
    """Deterministic stand-in for MediaPipe so tests need no model weights."""

    def __init__(self, landmarks=None, poses=None):
        if poses is not None:
            self.poses = [pose for pose in poses if pose]
            self.landmarks = self.poses[0] if self.poses else {}
        else:
            self.landmarks = (
                landmarks
                if landmarks is not None
                else {
                    "left_shoulder": loc(0.4, 0.3),
                    "left_elbow": loc(0.5, 0.45),
                    "left_wrist": loc(0.6, 0.6),
                }
            )
            self.poses = [self.landmarks] if self.landmarks else []
        self.frames = 0
        self.closed = False

    def process_frame(self, frame, timestamp_ms=None):
        self.frames += 1
        return list(self.poses)

    def close(self):
        self.closed = True


@pytest.fixture
def synthetic_video(tmp_path):
    """Write a short synthetic clip and return its path."""

    def _make(frames=6, size=(160, 120), fps=30.0, name="clip.mp4"):
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            pytest.skip("This OpenCV build cannot write mp4v video")
        try:
            for index in range(frames):
                frame = np.zeros((size[1], size[0], 3), np.uint8)
                frame[:, (index * 10) % size[0] :] = 255
                writer.write(frame)
        finally:
            writer.release()
        return path

    return _make
