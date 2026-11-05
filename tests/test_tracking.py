import pytest
from conftest import loc

from kinema.core.tracking import PersonTracker


def person(x, y=0.5):
    return {
        "left_hip": loc(x, y),
        "right_hip": loc(x + 0.05, y),
        "left_shoulder": loc(x, y - 0.2),
        "right_shoulder": loc(x + 0.05, y - 0.2),
    }


def test_first_detections_get_new_ids():
    tracker = PersonTracker(max_people=2)

    tracked = tracker.assign([person(0.2), person(0.7)])

    assert [entry.id for entry in tracked] == [1, 2]


def test_ids_survive_a_reordered_detection_list():
    tracker = PersonTracker(max_people=2)
    tracker.assign([person(0.2), person(0.7)])

    tracked = tracker.assign([person(0.71), person(0.21)])

    left_to_right = sorted(tracked, key=lambda entry: entry.landmarks["left_hip"].x)
    assert [entry.id for entry in left_to_right] == [1, 2]


def test_a_new_arrival_gets_a_new_id():
    tracker = PersonTracker(max_people=3)
    tracker.assign([person(0.2)])

    tracked = tracker.assign([person(0.21), person(0.8)])

    assert [entry.id for entry in tracked] == [1, 2]


def test_a_distant_detection_is_treated_as_a_new_person():
    tracker = PersonTracker(max_people=2, max_match_distance=0.05)
    tracker.assign([person(0.2)])

    tracked = tracker.assign([person(0.8)])

    assert [entry.id for entry in tracked] == [2]


def test_detections_are_capped_at_max_people():
    tracker = PersonTracker(max_people=2)

    tracked = tracker.assign([person(0.1), person(0.5), person(0.9)])

    assert len(tracked) == 2


def test_invalid_max_people_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        PersonTracker(max_people=0)
