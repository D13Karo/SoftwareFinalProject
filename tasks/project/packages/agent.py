import time
import random
import threading
import cv2
import numpy as np
import yaml
from enum import Enum, auto

with open("config/project_config.yaml", "r") as f:
    config = yaml.safe_load(f)

TAG_ID_MAP = {int(k): v for k, v in config["tag_id_map"].items()}

STOP_WAIT_SECONDS = config["thresholds"]["stop_wait_seconds"]
YIELD_WAIT_SECONDS = config["thresholds"]["yield_wait_seconds"]
# Closed-loop intersection turn: turn until the lane is re-acquired (centered).
TURN_MIN_SECONDS = config["thresholds"].get("turn_min_seconds", 0.4)
TURN_MAX_SECONDS = config["thresholds"].get("turn_max_seconds", 2.5)
TURN_ALIGN_STEER = config["thresholds"].get("turn_align_steer", 0.12)
# Crossing an intersection straight: the follower can't track the crossing
# markings, so drive straight until a clean lane reappears on the far side.
CROSS_MIN_SECONDS = config["thresholds"].get("cross_min_seconds", 0.5)
CROSS_MAX_SECONDS = config["thresholds"].get("cross_max_seconds", 2.5)
# Ignore tags smaller than this (apparent side length in px) so the bot reacts
# at the sign, not to a far/side sign across the map.
MIN_TAG_PIXELS = config["thresholds"].get("min_tag_pixels", 45)
# After a turn, ignore signs for this long so the bot drives clear of the
# intersection instead of turning again at the next turn sign (square loop).
SIGN_COOLDOWN = config["thresholds"].get("sign_cooldown_seconds", 8.0)
# Project-specific lane-follow tuning (sharp-curve friendly), separate from the
# visual_lane_servoing task's own config.
PROJECT_LANE_CONFIG = "config/project_lane_config.yaml"


class RobotState(Enum):
    LANE_FOLLOWING = auto()
    SIGN_DETECTED = auto()
    WAITING = auto()
    YIELD_CHECK = auto()
    RESUMING = auto()
    OBSTACLE_STOP = auto()


# --- Phase 4: intersection turns + LED state colors ---

# LEDs to drive (front-left, front-right, back-left, back-right); index 1 is unused.
LED_INDICES = [0, 2, 3, 4]

# Per-state color: green = driving, yellow = sign detected, red = stopped.
STATE_COLORS = {
    RobotState.LANE_FOLLOWING: [0.0, 1.0, 0.0],
    RobotState.SIGN_DETECTED:  [1.0, 1.0, 0.0],
    RobotState.WAITING:        [1.0, 0.0, 0.0],
    RobotState.YIELD_CHECK:    [1.0, 0.0, 0.0],
    RobotState.RESUMING:       [0.0, 1.0, 0.0],
    RobotState.OBSTACLE_STOP:  [1.0, 0.0, 0.0],
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

# (left, right) wheel speeds for the timed left/right turn across an intersection.
# "straight" is handled by lane following, not a blind drive.
TURN_SPEEDS = {
    "left":  (0.04, 0.16),
    "right": (0.16, 0.04),
}


def set_state_leds(leds, state):
    """Light all LEDs with the color for `state`. No-op when leds is None."""
    if not leds:
        return
    color = STATE_COLORS.get(state, [0.0, 0.0, 0.0])
    for i in LED_INDICES:
        leds.set_rgb(i, color)


_detector = None


def _get_detector():
    """Build the AprilTag detector once and reuse it.

    Constructing a Detector per frame leaks an os.add_dll_directory handle on
    Windows (crashes with WinError 206 after a few hundred frames) and is slow.
    """
    global _detector
    if _detector is None:
        from pupil_apriltags import Detector
        _detector = Detector()
    return _detector


def _tag_size(result):
    """Apparent size of a tag = its longest edge in pixels (grows as we approach)."""
    c = result.corners
    return max(float(np.linalg.norm(c[i] - c[(i + 1) % 4])) for i in range(4))


def detect_sign(frame):
    """Return (tag_id, sign) for the closest tag, but only once it's near enough.

    Picks the largest (closest) tag in view and ignores it while it's still far
    (below MIN_TAG_PIXELS), so the bot acts at the sign rather than from afar.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = _get_detector().detect(gray)

    if len(results) == 0:
        return None, None

    nearest = max(results, key=_tag_size)
    if _tag_size(nearest) < MIN_TAG_PIXELS:
        return None, None

    sign = TAG_ID_MAP.get(nearest.tag_id, "unknown")
    return nearest.tag_id, sign


_lane_agent = None
_lane_agent_failed = False


def _get_lane_agent():
    """Lazily build the visual lane-servoing agent, reused across frames.

    Imported lazily so importing this module never fails where the
    visual_lane_servoing package isn't available; falls back to driving
    straight if the agent can't be created.
    """
    global _lane_agent, _lane_agent_failed
    if _lane_agent is None and not _lane_agent_failed:
        try:
            from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
            _lane_agent = LaneServoingAgent(config_path=PROJECT_LANE_CONFIG)
        except Exception as e:
            print(f"[project] lane follower unavailable ({e}); driving straight")
            _lane_agent_failed = True
    return _lane_agent


def lane_follow(wheels, frame):
    """Steer to stay in the lane. `frame` is BGR (as returned by camera.read())."""
    agent = _get_lane_agent()
    if agent is None:
        wheels.set_wheels_speed(0.3, 0.3)
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    left, right = agent.compute_commands(rgb)
    wheels.set_wheels_speed(left, right)


def execute_turn(wheels, camera, turn, stop_event):
    """Closed-loop left/right turn across an intersection.

    Turns until the lane follower re-acquires a centered lane (|left-right| steer
    below TURN_ALIGN_STEER), capped at TURN_MAX_SECONDS. A minimum turn time first
    sweeps the bot off the entry lane so it doesn't immediately re-lock the lane it
    is leaving. Returns True if stop_event fired (caller should exit).
    """
    left, right = TURN_SPEEDS.get(turn, (0.0, 0.0))
    lane = _get_lane_agent()
    t0 = time.time()
    while True:
        if stop_event.is_set():
            return True
        wheels.set_wheels_speed(left, right)
        elapsed = time.time() - t0
        if elapsed >= TURN_MAX_SECONDS:
            return False
        if elapsed >= TURN_MIN_SECONDS and lane is not None:
            ok, frame = camera.read()
            if ok and frame is not None:
                l, r = lane.compute_commands(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if lane.last_debug_info.get("lane_detected") and abs(l - r) < TURN_ALIGN_STEER:
                    return False
        if stop_event.wait(0.05):
            return True


def cross_intersection(wheels, camera, stop_event):
    """Drive straight across an intersection until a clean lane is re-acquired.

    Inside a 4-way the yellow/white lines cross and confuse the lane follower, so
    we go straight (open loop) until the follower reports a centered lane on the
    far side, capped at CROSS_MAX_SECONDS. Returns True if stop_event fired.
    """
    lane = _get_lane_agent()
    t0 = time.time()
    good = 0
    while True:
        if stop_event.is_set():
            return True
        wheels.set_wheels_speed(0.13, 0.13)
        elapsed = time.time() - t0
        if elapsed >= CROSS_MAX_SECONDS:
            return False
        if elapsed >= CROSS_MIN_SECONDS and lane is not None:
            ok, frame = camera.read()
            if ok and frame is not None:
                l, r = lane.compute_commands(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                d = lane.last_debug_info
                # Only end the cross on a SOLID, centered lane held for several
                # frames, so a transient glimpse of the crossing markings inside the
                # intersection doesn't end it early (which veers the bot off-road).
                if (d.get("lane_detected") and d.get("total_lane_pixels", 0) > 1500
                        and abs(l - r) < TURN_ALIGN_STEER):
                    good += 1
                    if good >= 4:
                        return False
                else:
                    good = 0
        if stop_event.wait(0.05):
            return True


# --- obstacle stopping + right-of-way ---
# The object detector is heavy (CPU ONNX), so it runs in a BACKGROUND THREAD and
# only publishes flags. The control loop reads the flags instantly and never
# blocks on inference, so lane-following + sign detection stay responsive.
# Only ducks/trucks count as obstacles; signs (class 2) are handled via AprilTags.
OBSTACLE_CLASSES = {0, 1}
# Horizontal band (fraction of frame width) for "in my lane ahead"; obstacles
# whose center falls outside it (oncoming/side lanes) are ignored.
OBSTACLE_LANE_MIN = config["thresholds"].get("obstacle_lane_min", 0.33)
OBSTACLE_LANE_MAX = config["thresholds"].get("obstacle_lane_max", 0.72)
# Right-of-way: another robot/vehicle is class 1 ('truck'). It counts as cross
# traffic once its box is at least this big (px^2 in the model's square frame).
CROSS_TRAFFIC_CLASSES = {1}
CROSS_TRAFFIC_MIN_AREA = config["thresholds"].get("cross_traffic_min_area", 3500)

_obstacle_agent = None
_obstacle_failed = False
_should_stop = None

_det_lock = threading.Lock()
_obstacle_state = (False, "")    # (stop, reason), published by the detection thread
_cross_state = False             # cross-traffic present, published by the thread
_frame_lock = threading.Lock()
_shared_frame = None             # latest camera frame, shared with the detection thread


def _get_obstacle_agent():
    """Lazily build the object detector. Imported lazily so this module still
    imports where the object_detection package/model isn't available."""
    global _obstacle_agent, _obstacle_failed, _should_stop
    if _obstacle_agent is None and not _obstacle_failed:
        try:
            from tasks.object_detection.packages.agent import ObjectDetectionAgent
            from tasks.object_detection.packages.stop_activity import should_stop
            _should_stop = should_stop
            _obstacle_agent = ObjectDetectionAgent()
            if not _obstacle_agent.model_loaded:
                print(f"[project] obstacle model not loaded ({_obstacle_agent.load_error})")
        except Exception as e:
            print(f"[project] obstacle detection unavailable ({e}); not stopping for obstacles")
            _obstacle_failed = True
    return _obstacle_agent


def _publish_frame(frame):
    """Share the latest camera frame with the detection thread."""
    global _shared_frame
    with _frame_lock:
        _shared_frame = frame


def _detection_loop(stop_event):
    """Background thread: run the object detector on the latest frame and publish
    obstacle / cross-traffic flags. Heavy inference stays OUT of the control loop."""
    global _obstacle_state, _cross_state
    agent = _get_obstacle_agent()
    if agent is None or not agent.model_loaded:
        return                        # no model -> control loop just uses the defaults
    lo = OBSTACLE_LANE_MIN * agent.img_size
    hi = OBSTACLE_LANE_MAX * agent.img_size
    while not stop_event.is_set():
        with _frame_lock:
            frame = _shared_frame
        if frame is None:
            if stop_event.wait(0.03):
                break
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        square = cv2.resize(rgb, (agent.img_size, agent.img_size))
        dets = agent.detect(square)
        if dets is None:              # detector's own frame-skip
            continue
        obstacles = [d for d in dets
                     if d[2] in OBSTACLE_CLASSES and lo <= (d[0][0] + d[0][2]) / 2 <= hi]
        stop = _should_stop(obstacles, agent.img_size)
        cross = any(cls in CROSS_TRAFFIC_CLASSES and (x2 - x1) * (y2 - y1) >= CROSS_TRAFFIC_MIN_AREA
                    for (x1, y1, x2, y2), score, cls in dets)
        with _det_lock:
            _obstacle_state = stop
            _cross_state = cross


def obstacle_ahead():
    """Latest (stop, reason) from the detection thread (non-blocking)."""
    with _det_lock:
        return _obstacle_state


def cross_traffic():
    """Latest cross-traffic flag from the detection thread (non-blocking)."""
    with _det_lock:
        return _cross_state


def main(camera, wheels, leds, stop_event):
    state = RobotState.LANE_FOLLOWING
    detected_tag_id = None
    detected_sign = None
    handled_tag_id = None   # sign we already acted on; ignored until it leaves view
    clear_streak = 0        # consecutive "intersection clear" checks in YIELD_CHECK
    announced_yield = False
    resume_cooldown_until = 0.0   # ignore signs until this time after a sign action
    set_state_leds(leds, state)

    # Heavy object detection runs in the background so it never stalls the control
    # loop; lane-following + sign detection stay responsive on fresh frames.
    threading.Thread(target=_detection_loop, args=(stop_event,), daemon=True).start()

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                continue
            _publish_frame(frame)

            if state == RobotState.LANE_FOLLOWING:
                if stop_event.is_set():
                    break

                stop_now, reason = obstacle_ahead()
                if stop_now:
                    print(f"Obstacle: {reason}")
                    wheels.set_wheels_speed(0.0, 0.0)
                    state = RobotState.OBSTACLE_STOP
                    set_state_leds(leds, state)
                    continue

                # After acting on a sign, ignore signs briefly so the bot clears the
                # intersection instead of turning again at the next sign (square loop).
                if time.time() >= resume_cooldown_until:
                    tag_id, sign = detect_sign(frame)
                    if tag_id is None:
                        handled_tag_id = None        # sign left view; allow it (or any) again
                    elif tag_id != handled_tag_id:   # a new sign we haven't acted on yet
                        detected_tag_id = tag_id
                        detected_sign = sign
                        state = RobotState.SIGN_DETECTED
                        set_state_leds(leds, state)
                        continue

                lane_follow(wheels, frame)

            elif state == RobotState.SIGN_DETECTED:
                if stop_event.is_set():
                    break

                print(f"Detected: {detected_sign} (tag ID {detected_tag_id})")
                handled_tag_id = detected_tag_id   # don't re-fire on this sign

                if detected_sign == "stop":
                    wheels.set_wheels_speed(0.0, 0.0)
                    state = RobotState.WAITING
                    set_state_leds(leds, state)

                elif detected_sign == "yield":
                    wheels.set_wheels_speed(0.12, 0.12)
                    state = RobotState.WAITING
                    set_state_leds(leds, state)

                elif detected_sign in ALLOWED_TURNS:
                    # Signs list the legal turns; pick one at random (assignment spec).
                    turn = random.choice(ALLOWED_TURNS[detected_sign])
                    print(f"Intersection {detected_sign}: turning {turn}")
                    if turn != "straight" and execute_turn(wheels, camera, turn, stop_event):
                        break
                    # Ignore signs briefly so the bot clears this intersection instead
                    # of turning again at the next turn sign (square loop).
                    resume_cooldown_until = time.time() + SIGN_COOLDOWN
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

                clear_streak = 0
                announced_yield = False
                state = RobotState.YIELD_CHECK
                set_state_leds(leds, state)

            elif state == RobotState.YIELD_CHECK:
                if stop_event.is_set():
                    break

                # Right-of-way: stay stopped while another robot is crossing; the
                # one already in the intersection goes first. Proceed once clear.
                wheels.set_wheels_speed(0.0, 0.0)
                if cross_traffic():
                    if not announced_yield:
                        print("Yielding: letting another robot pass")
                        announced_yield = True
                    clear_streak = 0
                    if stop_event.wait(0.1):
                        break
                else:
                    clear_streak += 1
                    if clear_streak >= 3:           # clear for a few checks -> go
                        # Cross straight through the intersection (the follower can't
                        # track the crossing markings), then resume on the far side.
                        if cross_intersection(wheels, camera, stop_event):
                            break
                        resume_cooldown_until = time.time() + SIGN_COOLDOWN
                        detected_tag_id = None
                        detected_sign = None
                        state = RobotState.LANE_FOLLOWING
                        set_state_leds(leds, state)
                    elif stop_event.wait(0.05):
                        break

            elif state == RobotState.RESUMING:
                if stop_event.is_set():
                    break

                lane_follow(wheels, frame)

                if stop_event.wait(0.5):
                    break

                detected_tag_id = None
                detected_sign = None
                state = RobotState.LANE_FOLLOWING
                set_state_leds(leds, state)

            elif state == RobotState.OBSTACLE_STOP:
                if stop_event.is_set():
                    break

                wheels.set_wheels_speed(0.0, 0.0)
                stop_now, _ = obstacle_ahead()
                if not stop_now:
                    state = RobotState.LANE_FOLLOWING
                    set_state_leds(leds, state)
                elif stop_event.wait(0.1):
                    break

    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
