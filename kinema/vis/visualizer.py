import cv2
import numpy as np

from kinema.core.types import Location

COLOR_TEXT = (255, 255, 255)
COLOR_BG_BOX = (0, 0, 0)
PERSON_COLORS = (
    (245, 117, 66),  # blue-ish
    (66, 245, 117),  # green
    (66, 117, 245),  # orange
    (245, 66, 180),  # purple
)


class Visualizer:
    def __init__(self, min_visibility: float = 0.5):
        self.min_visibility = min_visibility

        # Colors (BGR)
        self.COLOR_BONE = PERSON_COLORS[0]
        self.COLOR_JOINT = (245, 66, 230)
        self.COLOR_TEXT = COLOR_TEXT
        self.COLOR_BG_BOX = COLOR_BG_BOX

        # Connections definition (pairs of landmark names)
        # Standard skeleton topology
        self.CONNECTIONS = [
            # Torso
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            # Arms
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            # Legs
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
            # Feet (Simple)
            ("left_ankle", "left_heel"),
            ("left_ankle", "left_foot_index"),
            ("right_ankle", "right_heel"),
            ("right_ankle", "right_foot_index"),
        ]

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: dict[str, Location],
        color: tuple[int, int, int] | None = None,
        label: str | None = None,
    ) -> np.ndarray:
        h, w, _ = frame.shape
        output = frame.copy()
        bone_color = color or self.COLOR_BONE

        # Helper to get pixel coords
        def to_pixel(loc: Location) -> tuple[int, int]:
            return int(loc.x * w), int(loc.y * h)

        # 1. Draw Bones
        for start_name, end_name in self.CONNECTIONS:
            if start_name in landmarks and end_name in landmarks:
                start = landmarks[start_name]
                end = landmarks[end_name]

                # Check visibility
                if (
                    start.visibility >= self.min_visibility
                    and end.visibility >= self.min_visibility
                ):
                    cv2.line(output, to_pixel(start), to_pixel(end), bone_color, 2)

        # 2. Draw Joints
        for _name, loc in landmarks.items():
            if loc.visibility >= self.min_visibility:
                cv2.circle(output, to_pixel(loc), 4, self.COLOR_JOINT, -1)

        # 3. Identify the person when more than one is tracked
        if label:
            anchors = [
                landmarks[name]
                for name in ("left_shoulder", "right_shoulder")
                if name in landmarks and landmarks[name].visibility >= self.min_visibility
            ]
            if anchors:
                center = anchors[0]
                px, py = int(center.x * w), int(center.y * h)
                cv2.putText(
                    output,
                    label,
                    (px - 12, py - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    bone_color,
                    2,
                    cv2.LINE_AA,
                )

        return output

    def draw_metrics(
        self,
        frame: np.ndarray,
        metrics: dict[str, float],
        landmarks: dict[str, Location],
        show_dashboard: bool = True,
    ) -> np.ndarray:
        h, w, _ = frame.shape
        output = frame.copy()

        # Helper for text
        def draw_text(img, text, pos, scale=0.6):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            # Background box
            cv2.rectangle(
                img, (pos[0], pos[1] - th - 5), (pos[0] + tw, pos[1] + 5), self.COLOR_BG_BOX, -1
            )
            cv2.putText(
                img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, self.COLOR_TEXT, 1, cv2.LINE_AA
            )

        # 1. Draw Angle Labels at Joints
        # Mapping metric name to joint location
        metric_loc_map = {
            "left_elbow_angle": "left_elbow",
            "right_elbow_angle": "right_elbow",
            "left_knee_angle": "left_knee",
            "right_knee_angle": "right_knee",
            "left_hip_angle": "left_hip",
            "right_hip_angle": "right_hip",
        }

        for metric_name, value in metrics.items():
            joint_name = metric_loc_map.get(metric_name)
            if joint_name and joint_name in landmarks:
                loc = landmarks[joint_name]
                if loc.visibility >= self.min_visibility:
                    px, py = int(loc.x * w), int(loc.y * h)
                    # Label format: "125°"
                    label = f"{int(value)}deg"
                    draw_text(output, label, (px + 10, py))

        # 2. Draw Dashboard (Top Left)
        if show_dashboard:
            y_offset = 30
            draw_text(output, "BIO-MECHANICS METRICS", (10, y_offset), scale=0.7)
            y_offset += 25

            for name, value in metrics.items():
                if name.endswith("_2d"):
                    continue
                short_name = (
                    name.replace("left_", "L.").replace("right_", "R.").replace("_angle", "")
                )
                text = f"{short_name}: {int(value)}"
                draw_text(output, text, (10, y_offset))
                y_offset += 20

        return output
