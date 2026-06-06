"""Lane-following + obstacle-aware agent.

The agent wraps the existing LaneServoingAgent and adds:

  * A YOLO ONNX detector that runs in a background thread so the control
    loop is never blocked by inference latency.
  * Frame-skipping driven by integration_activity.NUMBER_FRAMES_SKIPPED().
  * A two-state machine (LANE_FOLLOWING <-> OBSTACLE_PRESENT) that uses
    stop_activity.should_stop on the filtered detections.
  * A smooth ramp on the motor commands so the robot decelerates gradually
    when an obstacle appears and accelerates back gradually when it clears.
"""

import os
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.object_detection.packages import integration_activity, stop_activity
from tasks.object_detection.packages.yolo_detector import YoloDetector, Detection


_MODEL_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'models', 'best.onnx'
))
_CONFIG_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'object_detection_config.yaml'
))


def _load_config():
    try:
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


class ObjectDetectionAgent:
    """State machine: LANE_FOLLOWING <-> OBSTACLE_PRESENT."""

    STATE_LANE = "lane_following"
    STATE_STOP = "obstacle_present"

    def __init__(
        self,
        lane_agent: Optional[LaneServoingAgent] = None,
        model_path: str = _MODEL_PATH,
        image_size: Optional[int] = None,
        decel_rate: Optional[float] = None,
        accel_rate: Optional[float] = None,
        release_frames: Optional[int] = None,
        release_seconds: Optional[float] = None,
    ):
        cfg = _load_config()
        self.lane_agent = lane_agent or LaneServoingAgent()
        self.image_size = int(image_size if image_size is not None
                              else cfg.get('image_size', 640))
        self.decel_rate = float(decel_rate if decel_rate is not None
                                else cfg.get('decel_rate', 0.04))
        self.accel_rate = float(accel_rate if accel_rate is not None
                                else cfg.get('accel_rate', 0.03))
        self.release_frames = int(release_frames if release_frames is not None
                                  else cfg.get('release_frames', 3))
        # Once any stop signal has been seen, the agent stays in STOP until
        # the detector has produced *continuous* clearance for this many
        # seconds. Robust to flicker in YOLO confidence around the threshold.
        self.release_seconds = float(release_seconds if release_seconds is not None
                                     else cfg.get('release_seconds', 1.0))

        self.detector: Optional[YoloDetector] = None
        self._detector_error: Optional[str] = None
        try:
            self.detector = YoloDetector(model_path)
            print(f"[ObjectDetectionAgent] Loaded model: {model_path}")
        except Exception as e:
            # Don't crash the whole task if model is missing/broken —
            # fall back to plain lane following and surface the error.
            self._detector_error = str(e)
            print(f"[ObjectDetectionAgent] Detector disabled: {e}")

        # State
        self.state = self.STATE_LANE
        self._current_left = 0.0
        self._current_right = 0.0
        self._clear_streak = 0
        # Wall-clock timestamp of the last detector frame whose `should_stop`
        # was True. Used to keep STOP latched through brief flicker.
        self._last_stop_signal_time = 0.0

        # Shared with detector thread
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_idx = 0

        self._det_lock = threading.Lock()
        self._raw_detections: List[Detection] = []
        self._kept_detections: List[Detection] = []
        self._should_stop = False
        self._stop_reason = ""
        self._latency_ms = 0.0
        self._frames_processed = 0

        self._stop_event = threading.Event()
        if self.detector is not None:
            self._thread = threading.Thread(
                target=self._detector_loop, daemon=True, name="obj-det")
            self._thread.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Detection thread
    # ------------------------------------------------------------------

    def _detector_loop(self):
        last_idx = -1
        while not self._stop_event.is_set():
            skip = max(0, int(integration_activity.NUMBER_FRAMES_SKIPPED()))
            period = skip + 1

            with self._frame_lock:
                frame = self._latest_frame
                idx = self._frame_idx

            if frame is None or idx == last_idx:
                time.sleep(0.005)
                continue
            if (idx % period) != 0:
                last_idx = idx
                continue

            t0 = time.time()
            try:
                raw = self.detector.detect(frame)
            except Exception as e:
                print(f"[ObjectDetectionAgent] detect error: {e}")
                raw = []

            kept = [d for d in raw if self._passes_filters(d)]
            stop_now, reason = stop_activity.should_stop(kept, self.image_size)
            latency = (time.time() - t0) * 1000.0

            with self._det_lock:
                self._raw_detections = raw
                self._kept_detections = kept
                self._should_stop = stop_now
                self._stop_reason = reason
                self._latency_ms = latency
                self._frames_processed += 1

            # Diagnostic: when the model sees something but the filters drop
            # it, print exactly why so we can tune. Only logs the highest-score
            # raw detection to keep the terminal readable.
            if raw and not kept:
                bbox, score, cls = max(raw, key=lambda d: d[1])
                print(f"[det] RAW kept=0  cls={cls} score={score:.2f} "
                      f"bbox={bbox}  drops="
                      f"{self._why_dropped(bbox, score, cls)}")
            elif kept:
                bbox, score, cls = kept[0]
                print(f"[det] KEPT cls={cls} score={score:.2f} bbox={bbox} "
                      f"stop={stop_now}")

            last_idx = idx

    @staticmethod
    def _why_dropped(bbox, score, cls) -> str:
        reasons = []
        if not integration_activity.filter_by_classes(cls):
            reasons.append(f"class({cls})")
        if not integration_activity.filter_by_scores(score):
            reasons.append(f"score({score:.2f})")
        if not integration_activity.filter_by_bboxes(bbox):
            reasons.append(f"bbox{bbox}")
        return ",".join(reasons) or "stop_activity"

    @staticmethod
    def _passes_filters(det: Detection) -> bool:
        bbox, score, cls = det
        if not integration_activity.filter_by_classes(cls):
            return False
        if not integration_activity.filter_by_scores(score):
            return False
        if not integration_activity.filter_by_bboxes(bbox):
            return False
        return True

    # ------------------------------------------------------------------
    # Control loop (called every camera frame)
    # ------------------------------------------------------------------

    def compute_commands(self, image_rgb: np.ndarray) -> Tuple[float, float]:
        # Hand the *BGR* frame to the detector thread; lane_agent wants RGB.
        # Resize to 640x480 so bbox coordinates land in the coordinate system
        # that integration_activity (hardcoded 640) and stop_activity
        # (image_size=640) were written against. Godot streams 1280x720.
        if self.detector is not None:
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if bgr.shape[0] != 480 or bgr.shape[1] != 640:
                bgr = cv2.resize(bgr, (640, 480),
                                 interpolation=cv2.INTER_AREA)
            with self._frame_lock:
                self._latest_frame = bgr
                self._frame_idx += 1

        lane_l, lane_r = self.lane_agent.compute_commands(image_rgb)

        with self._det_lock:
            stop_now = self._should_stop

        now = time.time()
        if stop_now:
            self._last_stop_signal_time = now
            self._clear_streak = 0
            self.state = self.STATE_STOP
        else:
            self._clear_streak += 1
            # Stay latched in STOP until BOTH: enough main-loop frames clear
            # AND the detector has been continuously clear for release_seconds.
            time_since_stop = now - self._last_stop_signal_time
            if (self.state == self.STATE_STOP
                    and self._clear_streak >= self.release_frames
                    and time_since_stop >= self.release_seconds):
                self.state = self.STATE_LANE

        if self.state == self.STATE_STOP:
            target_l, target_r = 0.0, 0.0
            rate = self.decel_rate
        else:
            target_l, target_r = lane_l, lane_r
            rate = self.accel_rate

        self._current_left = _ramp(self._current_left, target_l, rate)
        self._current_right = _ramp(self._current_right, target_r, rate)
        return self._current_left, self._current_right

    # ------------------------------------------------------------------
    # Introspection (used by the visualization / web UI)
    # ------------------------------------------------------------------

    @property
    def info(self) -> dict:
        with self._det_lock:
            return {
                "state": self.state,
                "should_stop": self._should_stop,
                "stop_reason": self._stop_reason,
                "kept_detections": list(self._kept_detections),
                "raw_detections": list(self._raw_detections),
                "detector_latency_ms": self._latency_ms,
                "frames_processed": self._frames_processed,
                "detector_error": self._detector_error,
                "frames_skipped": int(integration_activity.NUMBER_FRAMES_SKIPPED()),
            }


def _ramp(current: float, target: float, max_step: float) -> float:
    delta = target - current
    if delta > max_step:
        return current + max_step
    if delta < -max_step:
        return current - max_step
    return target
