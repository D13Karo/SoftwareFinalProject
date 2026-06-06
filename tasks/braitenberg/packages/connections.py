from typing import Tuple
import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Left motor weight matrix: highest at bottom-left, decreasing toward top-right."""
    rows, cols = shape
    # row_w: 0 at top → 1 at bottom (closer objects are more urgent)
    row_w = np.linspace(0.0, 1.0, rows).reshape(-1, 1)
    # col_w: 1 at left col → 0 at right col
    col_w = np.linspace(1.0, 0.0, cols).reshape(1, -1)
    return row_w * col_w


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Right motor weight matrix: highest at bottom-right, decreasing toward top-left."""
    rows, cols = shape
    # row_w: 0 at top → 1 at bottom
    row_w = np.linspace(0.0, 1.0, rows).reshape(-1, 1)
    # col_w: 0 at left col → 1 at right col
    col_w = np.linspace(0.0, 1.0, cols).reshape(1, -1)
    return row_w * col_w
