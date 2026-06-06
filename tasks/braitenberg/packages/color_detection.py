import numpy as np
import cv2
from typing import Tuple


def detect_ducks(
    frame_rgb: np.ndarray,
    lower_hsv: np.ndarray,
    upper_hsv: np.ndarray,
) -> Tuple[float, float]:
    """
    Detect yellow ducks in the camera frame and return normalised sensor
    signals for the left and right halves of the image.

    The idea: split the frame down the middle. Count how many pixels in
    each half match the duck colour (using an HSV threshold mask). Divide
    by the total number of pixels in that half so the signal is always
    in [0, 1] regardless of image resolution.

    Args:
        frame_rgb:  RGB image from the camera driver (H x W x 3, uint8).
        lower_hsv:  Lower HSV bound for duck colour, shape (3,).
                    Note: OpenCV uses H in [0, 179], S and V in [0, 255].
        upper_hsv:  Upper HSV bound for duck colour, shape (3,).

    Returns:
        (left_signal, right_signal) – normalised duck pixel fraction for
        the left and right halves. Each value is in [0.0, 1.0].

    TODO:
        1. Convert frame_rgb to BGR, then to HSV with cv2.cvtColor.
        2. Create a binary mask with cv2.inRange(hsv, lower_hsv, upper_hsv).
        3. Split the mask in half (left columns / right columns).
        4. Compute the fraction of white (duck) pixels in each half.
        5. Return (left_signal, right_signal).

    Hint:
        half = frame_rgb.shape[1] // 2
        left_mask  = mask[:, :half]
        right_mask = mask[:, half:]
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)

    half = frame_rgb.shape[1] // 2
    left_mask  = mask[:, :half]
    right_mask = mask[:, half:]

    left_signal  = left_mask.sum()  / 255 / left_mask.size
    right_signal = right_mask.sum() / 255 / right_mask.size

    return left_signal, right_signal
