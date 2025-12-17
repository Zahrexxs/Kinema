import numpy as np
import math
from typing import Dict, Optional, List, Deque
from collections import deque
from ahma.core.pose_engine import Location

class OneEuroFilter:
    """
    Adaptive low-pass filter (One Euro Filter) for smoothing noisy signals.
    Minimizes jitter at low speeds while maintaining responsiveness at high speeds.
    """
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
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
    def __init__(self, fps: float = 30.0, smooth_landmarks: bool = True):
        self.fps = fps
        self.dt = 1.0 / fps
        self.use_filters = smooth_landmarks
        
        # Filters for each coordinate of each landmark: {landmark_name: {axis: OneEuroFilter}}
        self.filters: Dict[str, Dict[str, OneEuroFilter]] = {}
        self.current_time = 0.0
        
        # History for velocity calc: {landmark_name: deque([(time, value), ...])}
        self.velocity_history = {} # simple 2-frame delta for now

    def _get_filter(self, name: str, axis: str, initial_value: float) -> OneEuroFilter:
        if name not in self.filters:
            self.filters[name] = {}
        if axis not in self.filters[name]:
            # Parameters tuned for human motion (approximate)
            self.filters[name][axis] = OneEuroFilter(
                t0=self.current_time, 
                x0=initial_value, 
                min_cutoff=1.0, # 1 Hz cutoff at low speed (reduces jitter)
                beta=0.007,      # Responsiveness factor
                d_cutoff=1.0
            )
        return self.filters[name][axis]

    def update_landmarks(self, landmarks: Dict[str, Location]) -> Dict[str, Location]:
        """Apply smoothing to landmarks."""
        self.current_time += self.dt
        
        if not self.use_filters:
            return landmarks

        smoothed = {}
        for name, loc in landmarks.items():
            # Filter X
            f_x = self._get_filter(name, 'x', loc.x)
            new_x = f_x(self.current_time, loc.x)
            
            # Filter Y
            f_y = self._get_filter(name, 'y', loc.y)
            new_y = f_y(self.current_time, loc.y)
            
            # Filter Z
            f_z = self._get_filter(name, 'z', loc.z)
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

    def compute_metrics(self, landmarks: Dict[str, Location]) -> Dict[str, float]:
        """Compute standard biomechanical angles."""
        metrics = {}
        
        # Helper to safely get landmarks
        def get_loc(name):
            return landmarks.get(name)

        # 1. Elbow Angles (Flexion/Extension)
        # Left Elbow: L_Shoulder - L_Elbow - L_Wrist
        ls = get_loc('left_shoulder')
        le = get_loc('left_elbow')
        lw = get_loc('left_wrist')
        if ls and le and lw:
            metrics['left_elbow_angle'] = self.calculate_angle_3d(ls, le, lw)

        # Right Elbow
        rs = get_loc('right_shoulder')
        re = get_loc('right_elbow')
        rw = get_loc('right_wrist')
        if rs and re and rw:
            metrics['right_elbow_angle'] = self.calculate_angle_3d(rs, re, rw)

        # 2. Knee Angles
        # Left Knee: L_Hip - L_Knee - L_Ankle
        lh = get_loc('left_hip')
        lk = get_loc('left_knee')
        la = get_loc('left_ankle')
        if lh and lk and la:
            metrics['left_knee_angle'] = self.calculate_angle_3d(lh, lk, la)

        # Right Knee
        rh = get_loc('right_hip')
        rk = get_loc('right_knee')
        ra = get_loc('right_ankle')
        if rh and rk and ra:
            metrics['right_knee_angle'] = self.calculate_angle_3d(rh, rk, ra)

        # 3. Hip Angles (Trunk-Thigh)
        # Left Hip: L_Shoulder - L_Hip - L_Knee
        if ls and lh and lk:
            metrics['left_hip_angle'] = self.calculate_angle_3d(ls, lh, lk)
            
        if rs and rh and rk:
            metrics['right_hip_angle'] = self.calculate_angle_3d(rs, rh, rk)

        return metrics
