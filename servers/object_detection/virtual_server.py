"""Virtual server for the object_detection task.

Drives a state-machine-based lane-follower that pauses when the YOLO
detector reports a close obstacle. The detector runs in its own thread
inside ObjectDetectionAgent; this server only handles frame streaming,
the web UI, and start/stop/reset endpoints.
"""

import os
import sys
import socket
import threading

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

import cv2
from flask import Flask, Response, jsonify, render_template_string

from tasks.object_detection.packages.agent import ObjectDetectionAgent
from servers.object_detection.visualization import create_visualization
from servers.templates.object_detection import OBJECT_DETECTION_TEMPLATE

from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs


app = Flask(__name__)

camera = None
wheels = None
agent: ObjectDetectionAgent = None
running = False
stop_event = threading.Event()


def visualize(frame_rgb):
    """frame_rgb is RGB from the Godot camera."""
    global running
    if agent is None or wheels is None:
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    left, right = agent.compute_commands(frame_rgb)
    if running:
        wheels.set_wheels_speed(left, right)
    else:
        wheels.set_wheels_speed(0.0, 0.0)

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    # Match the resolution the detector worked in so bboxes line up.
    if bgr.shape[0] != 480 or bgr.shape[1] != 640:
        bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
    return create_visualization(bgr, left, right, agent.info)


generate_frames = make_frame_generator(lambda: camera, visualize, quality=55)


@app.route('/')
def index():
    return render_template_string(OBJECT_DETECTION_TEMPLATE,
                                  hostname=socket.gethostname())


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start():
    global running
    running = True
    print("[ObjectDetection] Started")
    return jsonify({'status': 'running'})


@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    print("[ObjectDetection] Stopped")
    return jsonify({'status': 'stopped'})


@app.route('/reset', methods=['POST'])
def reset():
    global agent
    if wheels is not None:
        wheels.reset_game()
    if agent is not None:
        agent._current_left = 0.0
        agent._current_right = 0.0
        agent.state = ObjectDetectionAgent.STATE_LANE
        agent.lane_agent._prev_error = 0.0
    return jsonify({'status': 'ok'})


@app.route('/running')
def get_running():
    return jsonify({'running': running})


@app.route('/status')
def status():
    if agent is None:
        return jsonify({'status': 'not_initialized'})
    info = agent.info
    return jsonify({
        'status': 'active',
        'state': info['state'],
        'should_stop': info['should_stop'],
        'stop_reason': info['stop_reason'],
        'detector_latency_ms': info['detector_latency_ms'],
        'frames_processed': info['frames_processed'],
        'frames_skipped': info['frames_skipped'],
        'kept_count': len(info['kept_detections']),
        'raw_count': len(info['raw_detections']),
        'detector_error': info['detector_error'],
        'running': running,
    })


def main():
    global camera, wheels, agent

    import argparse
    ap = argparse.ArgumentParser(description="Virtual Object Detection Server")
    ap.add_argument("--port",       type=int, default=5000)
    ap.add_argument("--frame-port", type=int, default=5001)
    ap.add_argument("--wheel-port", type=int, default=5002)
    ap.add_argument("--godot-host", type=str, default="localhost")
    args = ap.parse_args()

    suppress_http_logs()
    print("=" * 60)
    print("VIRTUAL OBJECT DETECTION SERVER")
    print("=" * 60)

    print("\n[1/3] Initializing wheels driver...")
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host,
        godot_port=args.wheel_port,
    )
    wheels.trim = 0

    print("\n[2/3] Initializing camera driver...")
    print(f"  Waiting for Godot on port {args.frame_port}...")
    camera = GodotCameraDriver(
        godot_config=GodotCameraConfig(host="0.0.0.0", port=args.frame_port))
    camera.start()
    print("  Camera: connected!")

    print("\n[3/3] Creating agent (loads ONNX model)...")
    agent = ObjectDetectionAgent()
    if agent._detector_error:
        print(f"  WARNING: detector disabled — {agent._detector_error}")
    print(f"  lane_agent: p_gain={agent.lane_agent.p_gain}, "
          f"d_gain={agent.lane_agent.d_gain}, "
          f"base_speed={agent.lane_agent.base_speed}")

    web_port = find_available_port(args.port)
    if web_port != args.port:
        print(f"  Port {args.port} busy, using {web_port}")

    print("\n" + "=" * 60)
    print(f"Web Interface: http://localhost:{web_port}")
    print("=" * 60 + "\n")

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if agent is not None:
            agent.shutdown()
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == "__main__":
    sys.exit(main())
