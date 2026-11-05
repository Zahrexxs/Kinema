from dataclasses import dataclass

from kinema.core.types import Location

# Hips and shoulders are the most stable parts of a pose, so they drive identity.
ANCHOR_LANDMARKS = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")


@dataclass(frozen=True)
class TrackedPerson:
    id: int
    landmarks: dict[str, Location]


class PersonTracker:
    """Assigns stable ids to per-frame detections.

    MediaPipe returns poses in an unstable order, so drawing and per-person
    metrics need an identity. Detections are matched greedily to the nearest
    previous centroid; a detection that matches nothing becomes a new id.
    Greedy nearest-neighbour can swap two ids when people cross, which is
    acceptable for per-frame metrics but not for long-term identity.
    """

    def __init__(self, max_match_distance: float = 0.2, max_people: int = 1) -> None:
        if max_people < 1:
            raise ValueError(f"max_people must be at least 1 (got {max_people!r})")
        self.max_match_distance = max_match_distance
        self.max_people = max_people
        self._centroids: dict[int, tuple[float, float]] = {}
        # Ids are 1-based so they read naturally next to P1/P2 labels.
        self._next_id = 1

    @staticmethod
    def centroid(landmarks: dict[str, Location]) -> tuple[float, float]:
        points = [landmarks[name] for name in ANCHOR_LANDMARKS if name in landmarks]
        if not points:
            points = list(landmarks.values())
        if not points:
            return 0.0, 0.0
        return (
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
        )

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def assign(self, poses: list[dict[str, Location]]) -> list[TrackedPerson]:
        detections = poses[: self.max_people]
        centroids = [self.centroid(pose) for pose in detections]

        candidates = sorted(
            (
                (self._distance(centroid, previous), track_id, index)
                for track_id, previous in self._centroids.items()
                for index, centroid in enumerate(centroids)
            ),
            key=lambda candidate: candidate[0],
        )

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignment: dict[int, int] = {}

        for distance, track_id, index in candidates:
            if distance > self.max_match_distance:
                break
            if track_id in matched_tracks or index in matched_detections:
                continue
            assignment[index] = track_id
            matched_tracks.add(track_id)
            matched_detections.add(index)

        tracked = []
        for index, landmarks in enumerate(detections):
            matched_id = assignment.get(index)
            if matched_id is None:
                matched_id = self._next_id
                self._next_id += 1
            tracked.append(TrackedPerson(id=matched_id, landmarks=landmarks))

        # Tracks that were not matched this frame are forgotten; a person who
        # disappears and returns gets a fresh id rather than a stale one.
        self._centroids = {person.id: centroids[index] for index, person in enumerate(tracked)}
        return tracked
