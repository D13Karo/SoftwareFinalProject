import time
import cv2
import yaml
from enum import Enum, auto

with open("config/project_config.yaml", "r") as f:
    config = yaml.safe_load(f)

TAG_ID_MAP = {int(k): v for k, v in config["tag_id_map"].items()}

STOP_WAIT_SECONDS = config["thresholds"]["stop_wait_seconds"]
YIELD_WAIT_SECONDS = config["thresholds"]["yield_wait_seconds"]


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

                if detected_sign == "stop":
                    wheels.set_wheels_speed(0.0, 0.0)

                elif detected_sign == "yield":
                    wheels.set_wheels_speed(0.12, 0.12)

                else:
                    state = RobotState.RESUMING
                    continue

                state = RobotState.WAITING

            elif state == RobotState.WAITING:
                if stop_event.is_set():
                    break

                if detected_sign == "stop":
                    wait_seconds = STOP_WAIT_SECONDS
                elif detected_sign == "yield":
                    wait_seconds = YIELD_WAIT_SECONDS
                else:
                    wait_seconds = 0

                if stop_event.wait(wait_seconds):
                    break

                state = RobotState.RESUMING

            elif state == RobotState.RESUMING:
                if stop_event.is_set():
                    break

                wheels.set_wheels_speed(0.2, 0.2)

                if stop_event.wait(0.5):
                    break

                detected_tag_id = None
                detected_sign = None
                state = RobotState.LANE_FOLLOWING

    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()