from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}

# ---------------------------------------------------------------------------
# Proximity thresholds
# ---------------------------------------------------------------------------
# The robot should stop when a duckie is *close* — not merely visible.
# "Closeness" is measured by two independent signals:
#
#   1. y2 (bottom edge of bbox) — a high y2 means the object is near the
#      bottom of the image, which corresponds to close in front of the bot.
#      Threshold: y2 > img_size * Y2_RATIO
#
#   2. Bounding-box area — a large box means the object is near.
#      Threshold: area > img_size² * AREA_RATIO
#
# Either condition alone is sufficient to trigger a stop.  Using both makes
# the system more robust to edge cases (e.g. a duckie at the very bottom
# corner of the frame, or a large duckie detected from the side).

Y2_RATIO   = 0.55   # stop when bottom edge is in the lower 45 % of frame
                    # (lowered from 0.70 — brakes sooner, more margin)
AREA_RATIO = 0.005  # stop when box covers ≥ 0.5 % of total frame area
                    # (lowered from 0.008 — earlier braking margin)


def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:
    """Decide whether the robot should stop based on the current detections.

    The function is called every processed frame (after frame-skipping) with
    the detections that survived all three filters in ``integration_activity``.

    Parameters
    ----------
    detections:
        List of ``((x1, y1, x2, y2), confidence, class_id)`` tuples.
        All entries have already passed ``filter_by_classes``,
        ``filter_by_scores``, and ``filter_by_bboxes``.
    img_size:
        Height (== width) of the camera frame in pixels.

    Returns
    -------
    (True,  reason_string)  → stop the robot and log the reason.
    (False, "")             → continue lane-following.

    Design notes
    ------------
    * We intentionally require the duckie to be *close*, not just detected.
      A duckie at the far end of the straight is not yet an obstacle.
    * If multiple duckies are visible, we stop as soon as *any one* of them
      crosses the proximity threshold.
    * The reason string is forwarded to the state-machine log so you can
      see in the terminal which condition triggered the stop.
    """
    if not detections:
        return False, ""

    frame_area = img_size * img_size

    for (x1, y1, x2, y2), score, cls_id in detections:
        obj_name = class_names.get(cls_id, f"class{cls_id}")

        # --- Proximity test 1: vertical position ---
        if y2 > img_size * Y2_RATIO:
            reason = (
                f"{obj_name} close ahead  "
                f"(y2={y2}, threshold={img_size * Y2_RATIO:.0f}, "
                f"conf={score:.2f})"
            )
            return True, reason

        # --- Proximity test 2: bounding-box area ---
        area = (x2 - x1) * (y2 - y1)
        if area > frame_area * AREA_RATIO:
            reason = (
                f"{obj_name} large box  "
                f"(area={area}, threshold={frame_area * AREA_RATIO:.0f}, "
                f"conf={score:.2f})"
            )
            return True, reason

    return False, ""
