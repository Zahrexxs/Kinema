import pytest

from kinema.core.metrics import MotionAnalyzer
from kinema.core.types import Location


def loc(x, y, z=0.0, visibility=1.0):
    return Location(x, y, z, visibility)


def test_angle_3d_right_angle():
    analyzer = MotionAnalyzer()

    angle = analyzer.calculate_angle_3d(loc(0, 1), loc(0, 0), loc(1, 0))

    assert angle == pytest.approx(90.0)


def test_angle_3d_straight_line():
    analyzer = MotionAnalyzer()

    angle = analyzer.calculate_angle_3d(loc(0, 1), loc(0, 0), loc(0, -1))

    assert angle == pytest.approx(180.0)


def test_angle_3d_ignores_zero_length_vector():
    analyzer = MotionAnalyzer()

    angle = analyzer.calculate_angle_3d(loc(0, 0), loc(0, 0), loc(1, 0))

    assert angle == 0.0


def test_angle_3d_45_degrees():
    analyzer = MotionAnalyzer()

    angle = analyzer.calculate_angle_3d(loc(1, 1), loc(0, 0), loc(1, 0))

    assert angle == pytest.approx(45.0)


def test_angle_2d_projection():
    analyzer = MotionAnalyzer()

    angle = analyzer.calculate_angle_2d(loc(0, 1), loc(0, 0), loc(1, 0))

    assert angle == pytest.approx(90.0)


def test_compute_metrics_returns_elbow_angle():
    analyzer = MotionAnalyzer()
    landmarks = {
        "left_shoulder": loc(0, 0),
        "left_elbow": loc(1, 0),
        "left_wrist": loc(1, 1),
    }

    metrics = analyzer.compute_metrics(landmarks)

    assert metrics == pytest.approx({"left_elbow_angle": 90.0})
    assert set(metrics) == {"left_elbow_angle"}


def test_compute_metrics_returns_all_six_angles():
    analyzer = MotionAnalyzer()
    landmarks = {
        "left_shoulder": loc(0, 0),
        "left_elbow": loc(1, 0),
        "left_wrist": loc(1, 1),
        "right_shoulder": loc(0, 0),
        "right_elbow": loc(-1, 0),
        "right_wrist": loc(-1, 1),
        "left_hip": loc(0, -1),
        "left_knee": loc(0, -2),
        "left_ankle": loc(0, -3),
        "right_hip": loc(0, -1),
        "right_knee": loc(0, -2),
        "right_ankle": loc(0, -3),
    }

    metrics = analyzer.compute_metrics(landmarks)

    assert set(metrics) == {
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
    }
    assert metrics["left_elbow_angle"] == pytest.approx(90.0)
    assert metrics["right_elbow_angle"] == pytest.approx(90.0)
    assert metrics["left_knee_angle"] == pytest.approx(180.0)
    assert metrics["right_knee_angle"] == pytest.approx(180.0)


def test_compute_metrics_without_joints_returns_empty():
    analyzer = MotionAnalyzer()

    assert analyzer.compute_metrics({"nose": loc(0.5, 0.5)}) == {}


def test_update_landmarks_preserves_names_and_visibility():
    analyzer = MotionAnalyzer(fps=30.0)

    smoothed = analyzer.update_landmarks({"left_elbow": loc(0.5, 0.5, visibility=0.9)})

    assert set(smoothed) == {"left_elbow"}
    assert smoothed["left_elbow"].visibility == 0.9
    assert smoothed["left_elbow"].x == pytest.approx(0.5)
    assert smoothed["left_elbow"].y == pytest.approx(0.5)


def test_update_landmarks_without_smoothing_returns_input():
    analyzer = MotionAnalyzer(smooth_landmarks=False)
    landmarks = {"left_elbow": loc(1, 2)}

    assert analyzer.update_landmarks(landmarks) is landmarks
