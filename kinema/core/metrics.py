import math

import numpy as np

from kinema.core.types import Location

DEFAULT_MIN_VISIBILITY = 0.5

# Each angle is measured at the middle landmark of the triplet.
ANGLE_TRIPLETS: dict[str, tuple[str, str, str]] = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
}


class OneEuroFilter:
    """
    Adaptive low-pass filter (One Euro Filter) for smoothing noisy signals.
    Minimizes jitter at low speeds while maintaining responsiveness at high speeds.
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev

    def __call__(self, t: float, x: float) -> float:
        t_e = t - self.t_prev

        # Prevent division by zero or negative time deltas
        if t_e <= 0:
            return self.x_prev

        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class MotionAnalyzer:
    def __init__(self, fps: float = 30.0, smooth_landmarks: bool = True) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive (got {fps!r})")
        self.fps = fps
        self.dt = 1.0 / fps
        self.use_filters = smooth_landmarks

        # Filters for each coordinate of each landmark: {landmark_name: {axis: OneEuroFilter}}
        self.filters: dict[str, dict[str, OneEuroFilter]] = {}
        self.current_time = 0.0

    def _get_filter(self, name: str, axis: str, initial_value: float) -> OneEuroFilter:
        if name not in self.filters:
            self.filters[name] = {}
        if axis not in self.filters[name]:
            # Parameters tuned for human motion (approximate)
            self.filters[name][axis] = OneEuroFilter(
                t0=self.current_time,
                x0=initial_value,
                min_cutoff=1.0,  # 1 Hz cutoff at low speed (reduces jitter)
                beta=0.007,  # Responsiveness factor
                d_cutoff=1.0,
            )
        return self.filters[name][axis]

    def update_landmarks(
        self, landmarks: dict[str, Location], t: float | None = None
    ) -> dict[str, Location]:
        """Apply smoothing to landmarks.

        ``t`` is the timestamp in seconds. Callers that know the true frame time
        should pass it; omitting it advances the clock by ``1 / fps``.
        """
        if t is None:
            self.current_time += self.dt
        else:
            self.current_time = t

        if not self.use_filters:
            return landmarks

        smoothed = {}
        for name, loc in landmarks.items():
            # Filter X
            f_x = self._get_filter(name, "x", loc.x)
            new_x = f_x(self.current_time, loc.x)

            # Filter Y
            f_y = self._get_filter(name, "y", loc.y)
            new_y = f_y(self.current_time, loc.y)

            # Filter Z
            f_z = self._get_filter(name, "z", loc.z)
            new_z = f_z(self.current_time, loc.z)

            smoothed[name] = Location(new_x, new_y, new_z, loc.visibility)

        return smoothed

    def calculate_angle_3d(self, a: Location, b: Location, c: Location) -> float:
        """
        Calculate the angle at point B formed by points A, B, C in 3D space.
        Returns degrees [0, 180].
        """
        # Vector BA
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        # Vector BC
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

        # Dot product
        dot_product = np.dot(ba, bc)

        # Magnitudes
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return 0.0

        cosine_angle = dot_product / (norm_ba * norm_bc)

        # Clamp for numerical stability
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def calculate_angle_2d(self, a: Location, b: Location, c: Location) -> float:
        """
        Calculate 2D angle (ignoring Z) at point B. Useful for projection on screen.
        """
        a_vec = np.array([a.x, a.y])
        b_vec = np.array([b.x, b.y])
        c_vec = np.array([c.x, c.y])

        ba = a_vec - b_vec
        bc = c_vec - b_vec

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def count_tracked(
        self, landmarks: dict[str, Location], min_visibility: float | None = None
    ) -> int:
        """Number of landmarks confident enough to be trusted."""
        threshold = DEFAULT_MIN_VISIBILITY if min_visibility is None else min_visibility
        return sum(1 for loc in landmarks.values() if loc.visibility >= threshold)

    def _triplet(
        self,
        landmarks: dict[str, Location],
        names: tuple[str, str, str],
        min_visibility: float,
    ) -> tuple[Location, Location, Location] | None:
        points = []
        for name in names:
            location = landmarks.get(name)
            if location is None or location.visibility < min_visibility:
                return None
            points.append(location)
        return points[0], points[1], points[2]

    def compute_metrics(
        self,
        landmarks: dict[str, Location],
        min_visibility: float | None = None,
    ) -> dict[str, float]:
        """Compute biomechanical angles for every fully visible joint chain.

        An angle is only reported when all three contributing landmarks clear
        ``min_visibility``; a landmark the model is unsure about would otherwise
        produce a confident-looking but meaningless number. Both the 3D angle and
        the image-plane (2D) projection are emitted.
        """
        threshold = DEFAULT_MIN_VISIBILITY if min_visibility is None else min_visibility

        metrics: dict[str, float] = {}
        for name, names in ANGLE_TRIPLETS.items():
            points = self._triplet(landmarks, names, threshold)
            if points is None:
                continue
            a, b, c = points
            metrics[name] = self.calculate_angle_3d(a, b, c)
            metrics[f"{name}_2d"] = self.calculate_angle_2d(a, b, c)

        return metrics
