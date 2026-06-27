import time
import random
import threading
import functools
import cv2
import numpy as np
import yaml
from enum import Enum, auto

# Flush every print immediately so the dashboard's piped log shows decisions
# live (Python block-buffers stdout when it isn't a TTY, e.g. on the robot).
print = functools.partial(print, flush=True)

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
# Project-specific lane-follow tuning. The real track and the sim's tight 90-deg
# curves need different speeds, so there are two profiles; main() picks the sim
# one automatically when it sees the Godot (simulation) wheel driver.
PROJECT_LANE_CONFIG     = "config/project_lane_config.yaml"       # real bot
PROJECT_LANE_SIM_CONFIG = "config/project_lane_sim_config.yaml"   # simulation
_lane_config_path = PROJECT_LANE_CONFIG


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
    """Light all LEDs with the color for `state`. No-op when leds is None.

    Wrapped in try/except: a flaky I2C LED bus (seen on some bots) must never
    crash the control loop.
    """
    if not leds:
        return
    color = STATE_COLORS.get(state, [0.0, 0.0, 0.0])
    try:
        for i in LED_INDICES:
            leds.set_rgb(i, color)
    except Exception:
        pass


# AprilTag detection backend, in order of preference:
#   1. pupil_apriltags (used in the sim)
#   2. OpenCV aruco 36h11 (if opencv-contrib is present)
#   3. built-in raw decoder (cv2 + numpy only) — for the robot, which has
#      neither library installed.
_detector = None
_detector_backend = None   # "pupil" | "aruco" | "raw" | "none"


def _get_detector():
    """Build the AprilTag detector once and reuse it.

    Tries pupil_apriltags first, then OpenCV's aruco 36h11 detector. Building a
    detector per frame leaks an os.add_dll_directory handle on Windows (crashes
    with WinError 206 after a few hundred frames) and is slow, so we cache the
    instance and the chosen backend.
    """
    global _detector, _detector_backend
    if _detector_backend is not None:
        return _detector

    try:
        from pupil_apriltags import Detector
        _detector = Detector(families="tag36h11")
        _detector_backend = "pupil"
        return _detector
    except Exception as e:
        print(f"[project] pupil_apriltags unavailable ({e}); trying OpenCV aruco")

    try:
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        _detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        _detector_backend = "aruco"
        return _detector
    except Exception as e:
        print(f"[project] OpenCV aruco unavailable ({e}); using built-in raw decoder")

    # Backend 3: dependency-free raw 36h11 decoder (cv2 + numpy only). Always
    # available, so sign detection works on the robot with no extra install.
    _detector_backend = "raw"
    _detector = None
    return _detector


def _edge_size(corners):
    """Apparent size of a tag = its longest edge in pixels (grows as we approach)."""
    c = np.asarray(corners, dtype=float)
    return max(float(np.linalg.norm(c[i] - c[(i + 1) % 4])) for i in range(4))


# --- dependency-free AprilTag 36h11 decoder (cv2 + numpy only) ---
# Canonical 36-bit codes for this project's tags, generated from the sim's own
# tag textures and verified against known-good 36h11 codes. The decoder tries
# all 4 rotations of each detected tag, so one code per tag is enough.
_CODES_36H11 = {
    0xd97f18b49: 1,
    0xdd280910e: 2,
    0xe479e9c98: 3,
    0xebcbca822: 4,
    0x22b1dfead: 8,
    0x265ad0472: 9,
    0x34fe91b86: 10,
    0x3ff962cd5: 11,
}
_DST_PTS = np.array([[0, 0], [79, 0], [79, 79], [0, 79]], dtype=np.float32)


def _order_corners(pts):
    """Sort 4 points into TL, TR, BR, BL order."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]])


def _decode_warped(bw80):
    """Sample the 8x8 grid of an 80x80 binary tag; return tag_id or None."""
    c = np.arange(8) * 10 + 5
    bits = bw80[np.ix_(c, c)]
    border = np.concatenate([bits[0, :], bits[7, :], bits[1:7, 0], bits[1:7, 7]])
    if np.any(border != 0):          # outer ring must be black
        return None
    inner = bits[1:7, 1:7]
    for k in range(4):               # try all 4 rotations
        code = int(np.rot90(inner, k).ravel().dot(1 << np.arange(35, -1, -1, dtype=np.int64)))
        if code in _CODES_36H11:
            return _CODES_36H11[code]
    return None


def _raw_decode(gray):
    """Detect 36h11 tags with only base OpenCV. Returns [(tag_id, corners)]."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    max_area = h * w * 0.5
    out, seen = [], set()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        src = _order_corners(approx.reshape(4, 2))
        M = cv2.getPerspectiveTransform(src, _DST_PTS)
        warped = cv2.warpPerspective(gray, M, (80, 80))
        _, bw = cv2.threshold(warped, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tid = _decode_warped(bw)
        if tid is not None and tid not in seen:
            seen.add(tid)
            out.append((tid, src))
    return out


def _raw_detections(frame):
    """Return [(tag_id, corners)] for every tag in `frame`, via the active backend."""
    _get_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if _detector_backend == "raw":
        return _raw_decode(gray)
    if _detector_backend == "aruco":
        corners, ids, _ = _detector.detectMarkers(gray)
        if ids is None:
            return []
        return [(int(i), c.reshape(-1, 2)) for c, i in zip(corners, ids.flatten())]
    if _detector_backend == "pupil":
        return [(int(r.tag_id), np.asarray(r.corners, dtype=float)) for r in _detector.detect(gray)]
    return []   # backend == "none"


# Temporal confirmation: require the same nearest tag for this many consecutive
# frames before acting, so a single noisy frame can't trigger a maneuver.
TAG_CONFIRM_FRAMES = config["thresholds"].get("tag_confirm_frames", 2)
_pending_tag = None
_pending_count = 0


def detect_sign(frame):
    """Return (tag_id, sign) for the closest tag once it's near enough AND has
    been seen for TAG_CONFIRM_FRAMES consecutive frames; otherwise (None, None).

    Picks the largest (closest) tag in view and ignores it while it's still far
    (below MIN_TAG_PIXELS), so the bot acts at the sign rather than from afar.
    """
    global _pending_tag, _pending_count

    nearest_id, nearest_size = None, 0.0
    for tag_id, corners in _raw_detections(frame):
        s = _edge_size(corners)
        if s > nearest_size:
            nearest_id, nearest_size = tag_id, s

    if nearest_id is None or nearest_size < MIN_TAG_PIXELS:
        _pending_tag, _pending_count = None, 0
        return None, None

    if nearest_id == _pending_tag:
        _pending_count += 1
    else:
        _pending_tag, _pending_count = nearest_id, 1

    if _pending_count < TAG_CONFIRM_FRAMES:
        return None, None

    return nearest_id, TAG_ID_MAP.get(nearest_id, "unknown")


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
            _lane_agent = LaneServoingAgent(config_path=_lane_config_path)
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
        if obstacle_ahead()[0]:
            wheels.set_wheels_speed(0.0, 0.0)
            return False    # obstacle: let the main loop's obstacle gate take over
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
        if obstacle_ahead()[0]:
            wheels.set_wheels_speed(0.0, 0.0)
            return False    # obstacle: let the main loop's obstacle gate take over
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


# --- peek left/right + left-priority yield (lets cross traffic pass) ---
# A forward-facing camera can't see the cross streets, so at a stop/yield the bot
# rotates to glance left, then right, watching for another robot each way. Left
# has priority: a vehicle on the left means we yield until it clears.
PEEK_TURN_LEFT       = config["thresholds"].get("peek_turn_left",  [-0.06, 0.06])
PEEK_TURN_RIGHT      = config["thresholds"].get("peek_turn_right", [0.06, -0.06])
PEEK_TURN_SECONDS    = config["thresholds"].get("peek_turn_seconds", 0.6)   # ~quarter turn
PEEK_HOLD_SECONDS    = config["thresholds"].get("peek_hold_seconds", 1.0)   # scan each side
LEFT_YIELD_TIMEOUT   = config["thresholds"].get("left_yield_timeout_seconds", 10.0)
VEHICLE_CLEAR_FRAMES = config["thresholds"].get("vehicle_clear_frames", 3)


def _peek_spin(wheels, camera, speeds, seconds, stop_event):
    """Spin in place for `seconds`, keeping the detector thread fed with frames.
    Returns "stopped" (stop_event), "obstacle", or None when finished."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        if stop_event.is_set():
            return "stopped"
        if obstacle_ahead()[0]:
            return "obstacle"
        wheels.set_wheels_speed(speeds[0], speeds[1])
        ok, frame = camera.read()
        if ok and frame is not None:
            _publish_frame(frame)
        if stop_event.wait(0.05):
            return "stopped"
    return None


def _peek_scan(wheels, camera, seconds, stop_event):
    """Hold still and watch for cross traffic for `seconds`.
    Returns (seen, status) where status is None, "stopped", or "obstacle"."""
    wheels.set_wheels_speed(0.0, 0.0)
    t0 = time.time()
    seen = False
    while time.time() - t0 < seconds:
        if stop_event.is_set():
            return seen, "stopped"
        if obstacle_ahead()[0]:
            return seen, "obstacle"
        ok, frame = camera.read()
        if ok and frame is not None:
            _publish_frame(frame)
        if cross_traffic():
            seen = True
        if stop_event.wait(0.05):
            return seen, "stopped"
    return seen, None


def peek_left_right(wheels, camera, stop_event):
    """Glance left, then right, watching for cross traffic each way.

    Returns "left" (vehicle on the left -> must yield, left has priority),
    "right"/"clear" (safe to go), "stopped" (stop_event), or "obstacle".
    """
    status = _peek_spin(wheels, camera, PEEK_TURN_LEFT, PEEK_TURN_SECONDS, stop_event)
    if status:
        return status
    left_seen, status = _peek_scan(wheels, camera, PEEK_HOLD_SECONDS, stop_event)
    if status:
        return status
    # swing right, past centre
    status = _peek_spin(wheels, camera, PEEK_TURN_RIGHT, PEEK_TURN_SECONDS * 2, stop_event)
    if status:
        return status
    right_seen, status = _peek_scan(wheels, camera, PEEK_HOLD_SECONDS, stop_event)
    if status:
        return status
    # re-centre
    status = _peek_spin(wheels, camera, PEEK_TURN_LEFT, PEEK_TURN_SECONDS, stop_event)
    if status:
        return status

    if left_seen:
        print("Peek: vehicle on the LEFT - yielding (left has priority)")
        return "left"
    if right_seen:
        print("Peek: vehicle on the RIGHT - we have priority, proceeding")
        return "right"
    print("Peek: no cross traffic - proceeding")
    return "clear"


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
    # The sim's tight 90-deg curves need a slower lane profile than the real
    # track; auto-select it from the Godot (simulation) wheel driver.
    global _lane_config_path
    if "godot" in type(wheels).__module__.lower():
        _lane_config_path = PROJECT_LANE_SIM_CONFIG
        print(f"[project] simulation detected -> lane profile: {_lane_config_path}")

    state = RobotState.LANE_FOLLOWING
    detected_tag_id = None
    detected_sign = None
    handled_tag_id = None   # sign we already acted on; ignored until it leaves view
    clear_streak = 0        # consecutive "intersection clear" checks in YIELD_CHECK
    announced_yield = False
    resume_cooldown_until = 0.0   # ignore signs until this time after a sign action
    yield_deadline = 0.0          # YIELD_CHECK: time after which we proceed anyway
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

            # Absolute-priority obstacle stop: interrupts every state (turns,
            # crossing, waiting, peeking) except OBSTACLE_STOP itself.
            if state != RobotState.OBSTACLE_STOP:
                stop_now, reason = obstacle_ahead()
                if stop_now:
                    print(f"Obstacle: {reason}")
                    wheels.set_wheels_speed(0.0, 0.0)
                    state = RobotState.OBSTACLE_STOP
                    set_state_leds(leds, state)
                    continue

            if state == RobotState.LANE_FOLLOWING:
                if stop_event.is_set():
                    break

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

                # Physically glance left, then right, for cross traffic before
                # entering the intersection (a forward-only check can't see the
                # cross streets). Left has priority.
                result = peek_left_right(wheels, camera, stop_event)
                if result == "stopped":
                    break
                if result == "obstacle":
                    wheels.set_wheels_speed(0.0, 0.0)
                    state = RobotState.OBSTACLE_STOP
                    set_state_leds(leds, state)
                    continue
                if result == "left":
                    # Vehicle on our left has priority — hold until it clears.
                    announced_yield = False
                    clear_streak = 0
                    yield_deadline = time.time() + LEFT_YIELD_TIMEOUT
                    state = RobotState.YIELD_CHECK
                    set_state_leds(leds, state)
                else:
                    # Right/clear — we may go. Cross, then resume on the far side.
                    if cross_intersection(wheels, camera, stop_event):
                        break
                    resume_cooldown_until = time.time() + SIGN_COOLDOWN
                    detected_tag_id = None
                    detected_sign = None
                    state = RobotState.LANE_FOLLOWING
                    set_state_leds(leds, state)

            elif state == RobotState.YIELD_CHECK:
                if stop_event.is_set():
                    break

                # Left has priority: stay stopped while a vehicle is still on the
                # left, up to LEFT_YIELD_TIMEOUT, then proceed once it clears.
                wheels.set_wheels_speed(0.0, 0.0)
                if cross_traffic() and time.time() < yield_deadline:
                    if not announced_yield:
                        print("Yielding: vehicle on the left has priority")
                        announced_yield = True
                    clear_streak = 0
                    if stop_event.wait(0.1):
                        break
                else:
                    clear_streak += 1
                    if clear_streak >= VEHICLE_CLEAR_FRAMES:
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
            try:
                leds.all_off()
            except Exception:
                pass
