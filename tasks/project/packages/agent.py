import time
import cv2
import yaml
from enum import Enum, auto

with open("config/project_config.yaml", "r") as f:
    config = yaml.safe_load(f)

TAG_ID_MAP = {int(k): v for k, v in config["tag_id_map"].items()}


class RobotState(Enum):
    LANE_FOLLOWING = auto()
    SIGN_DETECTED = auto()
    WAITING = auto()
    RESUMING = auto()

def detect_sign(frame):
    from pupil_apriltags import Detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = Detector()
    results = detector.detect(gray)
    if len(results) == 0:
        return None, None
    tag_id = results[0].tag_id
    sign = TAG_ID_MAP.get(tag_id, "unknown")
    return tag_id, sign

def main(camera, wheels, leds, stop_event):
    state = RobotState.LANE_FOLLOWING
    detected_tag_id = None
    detected_sign = None

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                continue

            if state == RobotState.LANE_FOLLOWING:
                if stop_event.is_set():
                    break

                tag_id, sign = detect_sign(frame)
                if tag_id is not None:
                    detected_tag_id = tag_id
                    detected_sign = sign
                    state = RobotState.SIGN_DETECTED
                    continue

                wheels.set_wheels_speed(0.3, 0.3)

            elif state == RobotState.SIGN_DETECTED:
                if stop_event.is_set():
                    break

                print(f"Detected: {detected_sign} (tag ID {detected_tag_id})")
                wheels.set_wheels_speed(0.0, 0.0)
                state = RobotState.WAITING

            elif state == RobotState.WAITING:
                if stop_event.is_set():
                    break

                if stop_event.wait(2.0):
                    break
                state = RobotState.RESUMING

            elif state == RobotState.RESUMING:
                if stop_event.is_set():
                    break

                wheels.set_wheels_speed(0.2, 0.2)
                if stop_event.wait(0.5):
                    break
                state = RobotState.LANE_FOLLOWING

    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()