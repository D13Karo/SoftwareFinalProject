import time
import cv2
import yaml

with open("config/project_config.yaml", "r") as f:
    config = yaml.safe_load(f)

TAG_ID_MAP = {int(k): v for k, v in config["tag_id_map"].items()}

def detect_sign(frame):
    import apriltag
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = apriltag.Detector()
    results = detector.detect(gray)
    if len(results) == 0:
        return None, None
    tag_id = results[0].tag_id
    sign = TAG_ID_MAP.get(tag_id, "unknown")
    return tag_id, sign

def main(camera, wheels, leds, stop_event):
    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                continue

            tag_id, sign = detect_sign(frame)
            if tag_id is not None:
                print(f"Detected: {sign} (tag ID {tag_id})")

    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()