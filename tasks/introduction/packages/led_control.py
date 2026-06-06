import colorsys
from typing import List


def set_driving_leds(left: float, right: float) -> dict:
    """
    Set LEDs to indicate driving direction.

    Duckiebot LED indices:
      0 = front-left
      2 = front-right
      3 = back-right
      4 = back-left

    Returns a dict mapping LED index -> [r, g, b] (values in [0.0, 1.0]).
    Front LEDs white when going forward, back LEDs red when going backward.
    """
    off   = [0.0, 0.0, 0.0]
    white = [1.0, 1.0, 1.0]
    red   = [1.0, 0.0, 0.0]

    avg = (left + right) / 2.0

    if avg > 0:
        return {0: white, 2: white, 3: off, 4: off}
    elif avg < 0:
        return {0: off,   2: off,   3: red, 4: red}
    else:
        return {0: off,   2: off,   3: off, 4: off}


def set_turning_leds(direction: str) -> dict:
    """
    Set LEDs to indicate turning direction.

    Duckiebot LED indices:
      0 = front-left
      2 = front-right
      3 = back-right
      4 = back-left

    Returns a dict mapping LED index -> [r, g, b] (values in [0.0, 1.0]).
    Left-side LEDs light orange for a left turn, right-side for right turn.
    All off for any other direction.
    """
    off    = [0.0, 0.0, 0.0]
    orange = [1.0, 0.5, 0.0]
    white  = [1.0, 1.0, 1.0]
    red    = [1.0, 0.0, 0.0]

    if direction == 'left':
        return {0: orange, 2: off,    3: off,    4: orange}
    elif direction == 'right':
        return {0: off,    2: orange, 3: orange, 4: off}
    elif direction == 'forward':
        return {0: white,  2: white,  3: off,    4: off}
    elif direction == 'stop':
        return {0: off,    2: off,    3: red,    4: red}
    else:
        return {0: off,    2: off,    3: off,    4: off}
