import numpy as np

from kinema.core.types import Location
from kinema.vis.visualizer import Visualizer


def loc(x, y, z=0.0, visibility=1.0):
    return Location(x, y, z, visibility)


LANDMARKS = {
    "left_shoulder": loc(0.4, 0.3),
    "left_elbow": loc(0.5, 0.5),
    "left_wrist": loc(0.6, 0.7),
}


def blank():
    return np.zeros((240, 320, 3), np.uint8)


def test_draw_skeleton_marks_pixels_without_mutating_the_input():
    frame = blank()

    drawn = Visualizer().draw_skeleton(frame, LANDMARKS)

    assert drawn.shape == frame.shape
    assert np.any(drawn != 0)
    assert not np.any(frame != 0)


def test_invisible_landmarks_are_not_drawn():
    faint = {name: loc(value.x, value.y, visibility=0.1) for name, value in LANDMARKS.items()}

    drawn = Visualizer().draw_skeleton(blank(), faint)

    assert not np.any(drawn != 0)


def test_visibility_threshold_is_configurable():
    faint = {name: loc(value.x, value.y, visibility=0.3) for name, value in LANDMARKS.items()}

    assert np.any(Visualizer(min_visibility=0.2).draw_skeleton(blank(), faint) != 0)
    assert not np.any(Visualizer(min_visibility=0.5).draw_skeleton(blank(), faint) != 0)


def test_draw_metrics_renders_labels():
    drawn = Visualizer().draw_metrics(blank(), {"left_elbow_angle": 90.0}, LANDMARKS)

    assert drawn.shape == (240, 320, 3)
    assert np.any(drawn != 0)


def test_draw_metrics_ignores_2d_duplicates():
    frame = blank()

    only_3d = Visualizer().draw_metrics(frame, {"left_elbow_angle": 90.0}, LANDMARKS)
    with_2d = Visualizer().draw_metrics(
        frame, {"left_elbow_angle": 90.0, "left_elbow_angle_2d": 12.0}, LANDMARKS
    )

    assert np.array_equal(only_3d, with_2d)


def test_draw_metrics_tolerates_missing_landmarks():
    drawn = Visualizer().draw_metrics(blank(), {"left_elbow_angle": 90.0}, {})

    assert np.any(drawn != 0)
