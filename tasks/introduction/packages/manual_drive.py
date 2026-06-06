from typing import Dict, Tuple
import logging
logger = logging.getLogger(__name__)

SPEED = 0.5
TURN = 0.3


def get_motor_speeds(keys_pressed: Dict[str, bool]) -> Tuple[float, float]:
    if keys_pressed.get('down'):
        return -SPEED, -SPEED

    left = 0.0
    right = 0.0

    if keys_pressed.get('up'):
        left  += SPEED
        right += SPEED
    if keys_pressed.get('left'):
        left  -= TURN
        right += TURN
    if keys_pressed.get('right'):
        left  += TURN
        right -= TURN

    left  = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))

    return left, right
