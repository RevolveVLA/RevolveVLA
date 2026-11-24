"""Where an expert picks a task up from the robot's current state (pure, no ROS): the phase to enter and the smoothness
measure of its first second. The entries are the task's phases: `approach` (fold, drive, reach the bar), `grasp` (the arm is
at the handle already: reach and close), `open` (the bar is held: the opening motion from the door's current angle)."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

APPROACH, GRASP, OPEN = "approach", "grasp", "open"
NEAR_M = 0.15           # TCP this close to the bar [m]: no fold, no drive (the pre-grasp is 0.10 back from it)
GRASPED_M = 0.03        # ... and closed hand this close: the bar is held
AJAR = 0.02             # door [rad] / drawer [m] opened this far counts as moved
CLOSED_FRACTION = 0.25  # the hand has travelled this part of the way from open to squeezing: it is closed


@dataclass(frozen=True)
class Situation:
    """What the entry depends on, measured at the moment of the takeover."""
    tcp_to_handle: float    # TCP to the handle bar (its box) at the current articulation [m]
    gripper: float          # the hand joint
    articulation: float     # door [rad] / drawer [m]
    phase: str | None = None    # the evaluator's phase, when an episode runs


def gripper_closed(gripper: float, case: dict) -> bool:
    return (case["hand_open"] - gripper) / (case["hand_open"] - case["hand_squeeze"]) >= CLOSED_FRACTION


def entry_for(s: Situation, case: dict) -> str:
    """The phase to enter: `open` when the hand is closed on the handle (or the evaluator is in its open phase with the hand
    closed), `grasp` when the TCP is near the handle, else `approach`."""
    closed = gripper_closed(s.gripper, case)
    if closed and (s.tcp_to_handle < GRASPED_M or s.phase == OPEN):
        return OPEN
    return GRASP if s.tcp_to_handle < NEAR_M else APPROACH


def is_ajar(s: Situation) -> bool:
    return s.articulation > AJAR


def max_delta_action(trace, t0: float, window_s: float = 1.0, hz: float = 30.0) -> float | None:
    """The largest change of any arm joint between two consecutive steps of a `hz` policy, over the `window_s` after `t0` [rad];
    `trace` = [(sim time, joint positions)] sorted by time, each step takes the last sample at or before it. None: no samples."""
    times = [t for t, _ in trace]
    steps = [bisect_right(times, t0 + k / hz) - 1 for k in range(int(window_s * hz) + 1)]
    rows = [trace[i][1] for i in steps if i >= 0]
    if len(rows) < 2:
        return None
    return max(abs(b - a) for p, q in zip(rows, rows[1:]) for a, b in zip(p, q))
