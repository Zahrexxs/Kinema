from types import SimpleNamespace

import numpy as np
import pytest

from kinema.core import pose_engine
from kinema.core.pose_engine import MediaPipeEstimator


def raw_landmarks(count=33):
    return [
        SimpleNamespace(x=index / 100, y=0.5, z=-0.1, visibility=0.9)
        for index in range(count)
    ]


class FakeLandmarker:
    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks
        self.timestamps = []
        self.image_calls = 0
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return SimpleNamespace(pose_landmarks=self.pose_landmarks)

    def detect(self, image):
        self.image_calls += 1
        return SimpleNamespace(pose_landmarks=self.pose_landmarks)

    def close(self):
        self.closed = True


@pytest.fixture
def make_estimator(monkeypatch, tmp_path):
    def _make(landmarks=None, static_image_mode=False):
        model = tmp_path / "pose_landmarker_full.task"
        model.write_bytes(b"stub")
        monkeypatch.setattr(
            pose_engine, "resolve_model_path", lambda base_dir, complexity: model
        )

        landmarker = FakeLandmarker(landmarks)

        class BaseOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class PoseLandmarkerOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class PoseLandmarker:
            @staticmethod
            def create_from_options(options):
                return landmarker

        vision = SimpleNamespace(
            RunningMode=SimpleNamespace(VIDEO="VIDEO", IMAGE="IMAGE"),
            PoseLandmarkerOptions=PoseLandmarkerOptions,
            PoseLandmarker=PoseLandmarker,
        )

        monkeypatch.setattr(pose_engine, "python", SimpleNamespace(BaseOptions=BaseOptions))
        monkeypatch.setattr(pose_engine, "vision", vision)

        return MediaPipeEstimator(static_image_mode=static_image_mode), landmarker

    return _make


def frame():
    return np.zeros((8, 8, 3), np.uint8)


def test_missing_model_raises_with_fetch_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pose_engine, "resolve_model_path", lambda base_dir, complexity: tmp_path / "nope.task"
    )

    with pytest.raises(FileNotFoundError, match="fetch_model.sh"):
        MediaPipeEstimator()


@pytest.mark.parametrize(
    ("index", "name"),
    [(0, "nose"), (11, "left_shoulder"), (23, "left_hip"), (27, "left_ankle"), (32, "right_foot_index")],
)
def test_landmark_names_cover_all_33_points(make_estimator, index, name):
    estimator, _ = make_estimator()

    assert estimator.landmark_names[index] == name
    assert len(estimator.landmark_names) == 33


def test_process_frame_maps_raw_landmarks(make_estimator):
    estimator, _ = make_estimator(landmarks=[raw_landmarks()])

    result = estimator.process_frame(frame())

    assert len(result) == 33
    assert result["nose"].x == pytest.approx(0.0)
    assert result["left_shoulder"].x == pytest.approx(0.11)
    assert result["right_foot_index"].visibility == pytest.approx(0.9)


def test_process_frame_returns_none_without_pose(make_estimator):
    estimator, _ = make_estimator(landmarks=None)

    assert estimator.process_frame(frame()) is None


def test_explicit_timestamps_are_used_and_stay_monotonic(make_estimator):
    estimator, landmarker = make_estimator(landmarks=[raw_landmarks()])

    estimator.process_frame(frame(), timestamp_ms=100)
    estimator.process_frame(frame(), timestamp_ms=250)
    estimator.process_frame(frame(), timestamp_ms=100)

    assert landmarker.timestamps == [100, 250, 251]


def test_missing_timestamp_falls_back_to_30fps(make_estimator):
    estimator, landmarker = make_estimator(landmarks=[raw_landmarks()])

    estimator.process_frame(frame())
    estimator.process_frame(frame())

    assert landmarker.timestamps == [33, 66]


def test_static_mode_uses_detect(make_estimator):
    estimator, landmarker = make_estimator(
        landmarks=[raw_landmarks()], static_image_mode=True
    )

    estimator.process_frame(frame())

    assert landmarker.image_calls == 1
    assert landmarker.timestamps == []


def test_context_manager_closes_estimator(make_estimator):
    estimator, landmarker = make_estimator()

    with estimator as entered:
        assert entered is estimator

    assert landmarker.closed is True
