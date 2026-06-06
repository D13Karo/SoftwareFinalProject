from typing import Tuple


def get_wheel_speeds(
    left_signal: float,
    right_signal: float,
    gain: float,
    const: float,
) -> Tuple[float, float]:
    """
    Compute wheel speeds using the Braitenberg Vehicle 2b avoidance rule.

    A Braitenberg Vehicle 2b ("fear") cross-wires each sensor to the
    *opposite* wheel with an excitatory connection:

        left_wheel  = const + gain * right_signal
        right_wheel = const + gain * left_signal

    Effect: when a duck appears on the LEFT (high left_signal), the RIGHT
    wheel speeds up → robot turns LEFT, steering AWAY from the duck, and
    vice versa.

    The `const` term sets the baseline forward speed so the robot keeps
    moving when no ducks are visible.

    Args:
        left_signal:  Duck detection signal from the left half  [0, 1].
        right_signal: Duck detection signal from the right half [0, 1].
        gain:         How strongly a detected duck steers the robot.
                      Start around 0.5–1.0 and tune from the dashboard.
        const:        Baseline (forward) speed for both wheels.
                      Start around 0.2–0.4.

    Returns:
        (left_speed, right_speed) – wheel speeds clamped to [-1.0, 1.0].

    TODO:
        1. Implement the cross-wired formula above.
        2. Clamp both outputs to [-1.0, 1.0] before returning.

    Hint:
        Use max(-1.0, min(1.0, value)) to clamp, or np.clip.
    """
    left  = const + gain * right_signal
    right = const + gain * left_signal

    left  = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))

    return left, right
