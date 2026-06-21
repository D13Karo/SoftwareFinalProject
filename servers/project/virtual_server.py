import sys
import os
import threading
import argparse

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

import numpy as np
import cv2
from flask import Flask, Response, jsonify, request

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.led_driver.virtual_led_driver import VirtualLEDsDriver
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs
from servers.templates.project import get_template

import tasks.project.packages.agent as agent
from tasks.project.packages.agent import TAG_ID_MAP


app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None
stop_event = threading.Event()

# --- overlay / status state (the agent runs in its own thread; we surface it here) ---
OVERLAY_EVERY    = 2          # only re-detect every Nth frame to keep CPU sane
_detector        = None
_frame_idx       = 0
_cached_dets     = []         # last drawn detections: [(corners, tag_id, sign)]
_last_sign       = None
_last_tag_id     = None
_overlay_lock    = threading.Lock()


def _get_detector():
    """Lazily build one reusable AprilTag detector for the dashboard overlay."""
    global _detector
    if _detector is None:
        try:
            from pupil_apriltags import Detector
            _detector = Detector(families="tag36h11")
        except Exception as e:
            print(f"[project sim] AprilTag overlay disabled ({e})")
            _detector = False  # sentinel: tried and failed
    return _detector or None


def _state_label(r, g, b):
    """Map the agent's LED state color back to a human label (see agent.STATE_COLORS)."""
    if r > 0.5 and g > 0.5:
        return "SIGN DETECTED"
    if r > 0.5:
        return "STOPPED / WAITING"
    if g > 0.5:
        return "DRIVING"
    return "IDLE"


def _draw_state_banner(bgr):
    if leds is None:
        return
    r, g, b = leds.get_state(0)
    color = (int(b * 255), int(g * 255), int(r * 255))  # LED RGB -> cv2 BGR
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 28), color, -1)
    cv2.putText(bgr, _state_label(r, g, b), (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def _placeholder():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for Godot camera...", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return blank


def visualize(frame):
    """frame is BGR (make_frame_generator rgb=False). Overlay tags + agent state."""
    global _frame_idx, _cached_dets, _last_sign, _last_tag_id

    if frame is None:
        return _placeholder()

    bgr = frame
    _frame_idx += 1

    det = _get_detector()
    if det is not None and (_frame_idx % OVERLAY_EVERY == 0):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        try:
            results = det.detect(gray)
        except Exception:
            results = []
        dets = [(r.corners.astype(int), r.tag_id, TAG_ID_MAP.get(r.tag_id, "unknown"))
                for r in results]
        _cached_dets = dets
        if dets:
            with _overlay_lock:
                _last_tag_id = int(dets[0][1])
                _last_sign   = dets[0][2]

    for corners, tag_id, sign in _cached_dets:
        cv2.polylines(bgr, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        x, y = corners[0]
        cv2.putText(bgr, f"{sign} #{tag_id}", (int(x), max(12, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    _draw_state_banner(bgr)
    return bgr


generate_frames = make_frame_generator(lambda: camera, visualize, quality=60, rgb=False)


@app.route('/')
def index():
    return get_template(title='Project — Traffic Signs', subtitle='Simulation')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with _overlay_lock:
        last_sign, last_tag = _last_sign, _last_tag_id
    led_r, led_g, led_b = leds.get_state(0) if leds else (0.0, 0.0, 0.0)
    return jsonify({
        'running':      not stop_event.is_set(),
        'state':        _state_label(led_r, led_g, led_b),
        'last_sign':    last_sign,
        'last_tag_id':  last_tag,
        'game_over':    wheels.is_game_over() if wheels else False,
        'leds':         leds.get_all_states() if leds else {},
    })


@app.route('/command', methods=['POST'])
def command():
    data = request.json or {}
    key = (data.get('key') or '').strip().lower()
    if key == 'reset' and wheels:
        wheels.reset_game()
        return jsonify({'status': 'ok', 'message': 'game reset'})
    if not key:
        return jsonify({'status': 'error', 'message': 'key required'}), 400
    return jsonify({'status': 'ok', 'message': f'ignored {key!r} (no handler)'})


@app.route('/shutdown')
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({'status': 'ok'})


def main():
    global camera, wheels, leds, stop_event

    ap = argparse.ArgumentParser(description='Project Server — Simulation')
    ap.add_argument('--port',       type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT SERVER — SIMULATION (TRAFFIC SIGNS)')
    print('=' * 60)

    print('\n[1/4] Initializing virtual LEDs...')
    leds = VirtualLEDsDriver(debug=False)
    leds.all_off()
    print('  LEDs: ok')

    print('\n[2/4] Initializing Godot wheels...')
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host, godot_port=args.wheel_port,
    )

    print('\n[3/4] Initializing Godot camera...')
    camera = GodotCameraDriver(godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port))
    camera.start()
    print('  Camera: ok')

    print('\n[4/4] Starting agent...')
    stop_event.clear()
    threading.Thread(
        target=agent.main,
        args=(camera, wheels, leds, stop_event),
        daemon=True,
        name='AgentThread',
    ).start()
    print('  agent.main() running')

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://localhost:{web_port}')
    print('=' * 60 + '\n')

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
