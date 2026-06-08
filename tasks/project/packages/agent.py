import time
import random
import cv2
import yaml
from enum import Enum, auto

with open("config/project_config.yaml", "r") as f:
    config = yaml.safe_load(f)

TAG_ID_MAP = {int(k): v for k, v in config["tag_id_map"].items()}

STOP_WAIT_SECONDS = config["thresholds"]["stop_wait_seconds"]
YIELD_WAIT_SECONDS = config["thresholds"]["yield_wait_seconds"]
TURN_DURATION_SECONDS = config["thresholds"]["turn_duration_seconds"]


class RobotState(Enum):
    LANE_FOLLOWING = auto()
    SIGN_DETECTED = auto()
    WAITING = auto()
    RESUMING = auto()


# --- Phase 4: intersection turns + LED state colors ---

# LEDs to drive (front-left, front-right, back-left, back-right); index 1 is unused.
LED_INDICES = [0, 2, 3, 4]

# Per-state color: green = driving, yellow = sign detected, red = stopped.
STATE_COLORS = {
    RobotState.LANE_FOLLOWING: [0.0, 1.0, 0.0],
    RobotState.SIGN_DETECTED:  [1.0, 1.0, 0.0],
    RobotState.WAITING:        [1.0, 0.0, 0.0],
    RobotState.RESUMING:       [0.0, 1.0, 0.0],
}

# Allowed turns per intersection sign. Tune to match the course's intersection rules.
ALLOWED_TURNS = {
    "no-right-turn":     ["straight", "left"],
    "no-left-turn":      ["straight", "right"],
    "4-way-intersect":   ["straight", "left", "right"],
    "right-T-intersect": ["straight", "right"],
    "left-T-intersect":  ["straight", "left"],
    "T-intersection":    ["left", "right"],
}

# (left, right) wheel speeds for each timed turn primitive.
TURN_SPEEDS = {
    "straight": (0.25, 0.25),
    "left":     (0.05, 0.30),
    "right":    (0.30, 0.05),
}


def set_state_leds(leds, state):
    """Light all LEDs with the color for `state`. No-op when leds is None."""
    if not leds:
        return
    color = STATE_COLORS.get(state, [0.0, 0.0, 0.0])
    for i in LED_INDICES:
        leds.set_rgb(i, color)


def execute_turn(wheels, turn, stop_event):
    """Drive a timed turn. Returns True if stop_event fired (caller should exit)."""
    left, right = TURN_SPEEDS.get(turn, (0.0, 0.0))
    wheels.set_wheels_speed(left, right)
    return stop_event.wait(TURN_DURATION_SECONDS)


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
    set_state_leds(leds, state)

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
                    set_state_leds(leds, state)
                    continue

                wheels.set_wheels_speed(0.3, 0.3)

            elif state == RobotState.SIGN_DETECTED:
                if stop_event.is_set():
                    break

                print(f"Detected: {detected_sign} (tag ID {detected_tag_id})")

                if detected_sign == "stop":
                    wheels.set_wheels_speed(0.0, 0.0)
                    state = RobotState.WAITING
                    set_state_leds(leds, state)

                elif detected_sign == "yield":
                    wheels.set_wheels_speed(0.12, 0.12)
                    state = RobotState.WAITING
                    set_state_leds(leds, state)

                elif detected_sign in ALLOWED_TURNS:
                    turn = random.choice(ALLOWED_TURNS[detected_sign])
                    print(f"Intersection {detected_sign}: turning {turn}")
                    if execute_turn(wheels, turn, stop_event):
                        break
                    state = RobotState.RESUMING
                    set_state_leds(leds, state)

                else:
                    state = RobotState.RESUMING
                    set_state_leds(leds, state)

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
                set_state_leds(leds, state)

            elif state == RobotState.RESUMING:
                if stop_event.is_set():
                    break

                wheels.set_wheels_speed(0.2, 0.2)

                if stop_event.wait(0.5):
                    break

                detected_tag_id = None
                detected_sign = None
                state = RobotState.LANE_FOLLOWING
                set_state_leds(leds, state)

    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
