from typing import Tuple

MODEL_PATH = "tasks/object_detection/models/best.onnx"


def NUMBER_FRAMES_SKIPPED() -> int:
    return 1


def filter_by_classes(pred_class: int) -> bool:
    # Accept duckies and trucks (another duckiebot). The model sometimes
    # classifies a very-close, large-scale duck as 'truck' rather than
    # 'duckie' — keeping both makes the stop more reliable.
    return pred_class in (0, 1)


def filter_by_scores(score: float) -> bool:
    return score >= 0.35  # lowered from 0.6 — close-up ducks have flickery confidence


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    xmin, ymin, xmax, ymax = bbox
    area = (xmax - xmin) * (ymax - ymin)
    if area < 1800:          # lowered from 2500 — catch ducks slightly earlier
        return False
    cx = (xmin + xmax) / 2
    # Loosened lane guard: 0.10..0.90 instead of 0.20..0.80 so duckies on the
    # outside of a curve still trigger a stop while the bot's heading swings.
    if cx < 640 * 0.10 or cx > 640 * 0.90:
        return False
    # Lowered: 0.30 instead of 0.40 so detection fires sooner on approach.
    if ymax < 640 * 0.30:
        return False
    return True