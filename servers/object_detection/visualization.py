"""Composite visualization for the object_detection task."""

import cv2
import numpy as np

from tasks.object_detection.packages.dataset_activity import CLASSES

_CLASS_NAMES = {i: name for i, name in enumerate(CLASSES)}
_CLASS_COLOR = {
    0: (0, 215, 255),   # duckie -> orange
    1: (50, 50, 255),   # truck  -> red
    2: (255, 200, 0),   # sign   -> blue
}
_DROPPED_COLOR = (110, 110, 110)


def create_visualization(
    bgr: np.ndarray,
    pwm_left: float,
    pwm_right: float,
    info: dict,
) -> np.ndarray:
    h, w = bgr.shape[:2]
    canvas = bgr.copy()

    for bbox, score, cls in info.get("raw_detections", []):
        if (bbox, score, cls) in info.get("kept_detections", []):
            continue
        _draw_box(canvas, bbox, _DROPPED_COLOR,
                  f"{_CLASS_NAMES.get(cls, cls)} {score:.2f}", dashed=True)

    for bbox, score, cls in info.get("kept_detections", []):
        color = _CLASS_COLOR.get(cls, (0, 255, 0))
        _draw_box(canvas, bbox, color,
                  f"{_CLASS_NAMES.get(cls, cls)} {score:.2f}")

    overlay = _info_strip(w, pwm_left, pwm_right, info)
    return np.vstack([canvas, overlay])


def _draw_box(canvas, bbox, color, label, dashed=False):
    x1, y1, x2, y2 = bbox
    if dashed:
        _dashed_rect(canvas, (x1, y1), (x2, y2), color, 1, dash=6)
    else:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
    ty = max(0, y1 - 4)
    cv2.rectangle(canvas, (x1, ty - th - 2), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(canvas, label, (x1 + 2, ty),
                font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def _dashed_rect(canvas, p1, p2, color, thickness, dash=6):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(canvas, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(canvas, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(canvas, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(canvas, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _info_strip(width, pwm_left, pwm_right, info):
    h = 130
    canvas = np.zeros((h, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    state = info.get("state", "?")
    is_stopped = state == "obstacle_present"
    state_color = (0, 80, 240) if is_stopped else (60, 200, 80)
    cv2.putText(canvas, f"STATE: {state.upper()}", (12, 24),
                font, 0.6, state_color, 2, cv2.LINE_AA)

    err = info.get("detector_error")
    if err:
        cv2.putText(canvas, f"DETECTOR DISABLED: {err[:60]}", (12, 50),
                    font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        reason = info.get("stop_reason", "") or "—"
        cv2.putText(canvas, f"reason: {reason[:80]}", (12, 50),
                    font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    line = (f"det: {info.get('frames_processed', 0):d}  "
            f"latency: {info.get('detector_latency_ms', 0):.0f} ms  "
            f"skip: {info.get('frames_skipped', 0)}  "
            f"kept: {len(info.get('kept_detections', []))}/"
            f"{len(info.get('raw_detections', []))}")
    cv2.putText(canvas, line, (12, 75), font, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    _draw_pwm_bar(canvas, "L", 12, 90, width // 2 - 24, pwm_left)
    _draw_pwm_bar(canvas, "R", width // 2 + 12, 90, width // 2 - 24, pwm_right)
    return canvas


def _draw_pwm_bar(canvas, label, x, y, w, value):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar_h = 14
    cv2.putText(canvas, label, (x, y + 12), font, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    bx = x + 20
    bw = w - 70
    cv2.rectangle(canvas, (bx, y), (bx + bw, y + bar_h), (50, 50, 50), -1)
    fill = int(bw * max(0.0, min(1.0, abs(value))))
    color = (100, 200, 255) if value >= 0 else (255, 100, 100)
    cv2.rectangle(canvas, (bx, y), (bx + fill, y + bar_h), color, -1)
    cv2.putText(canvas, f"{value:+.2f}", (bx + bw + 6, y + 12),
                font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
