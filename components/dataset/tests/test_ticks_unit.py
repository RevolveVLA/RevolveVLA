"""dataset.ticks, the parts that need no bag: where Rosetta's porter lays its ticks (start + i x step, from the bag's
starting_time), how many emitted frames of an action stream are zero-filled because its first message comes late, and
of_episodes turning a bag rosbag2 cannot read into an `inputs:` problem before anything is ported."""
from types import SimpleNamespace

import pytest

from dataset import ticks

STEP_30 = int(1e9 / 30)          # Rosetta's step_ns(30) (rosetta/frames/resample.py)


@pytest.mark.parametrize("start, step, warmup, first, want", [
    (0, 10, 2, 20, 0),            # the first message exactly at the first emitted tick (20)
    (0, 10, 2, 21, 1),
    (0, 10, 2, 45, 3),            # ticks 20, 30, 40
    (0, 10, 2, 30, 1),            # a message at a tick counts for that tick: only tick 20 is late
    (0, 10, 2, 5, 0),             # before the first emitted tick: never negative
    (0, 10, 2, -50, 0),           # before the bag's start
    (0, 10, 0, 1, 1),             # no warmup: tick 0 itself is late
    (1_000, 33, 3, 1_100, 1),     # a non-zero start, emitted from tick 3 (1099 < 1100)
])
def test_late_action_frames(start, step, warmup, first, want):
    # emitted frames (ticks i >= warmup) whose time start + i x step is strictly before the stream's first message
    got = ticks.late_action_frames(start, step, warmup, first)
    assert got == want and type(got) is int


def test_late_action_frames_on_the_porter_grid():
    # a gripper stream whose first message lands 110 ms after the first emitted frame at 30 fps: 4 zero-filled frames
    start, warmup = 1_000_000_000, 2
    first_emitted = start + warmup * STEP_30
    assert ticks.late_action_frames(start, STEP_30, warmup, first_emitted + 110_000_000) == 4
    assert ticks.late_action_frames(start, STEP_30, warmup, first_emitted) == 0
    assert ticks.late_action_frames(start, STEP_30, warmup, first_emitted + 1) == 1


def test_frames_are_the_ticks_after_warmup():
    assert ticks.Ticks(603, 2).frames == 601
    assert ticks.Ticks(5, 5).frames == 0                    # every tick before the last observation stream starts
    assert ticks.Ticks(603, 2).action_late == {}


def test_of_episodes_unreadable_bag_is_an_input_problem(monkeypatch):
    # every bag is read before the port: one rosbag2 cannot open is an `inputs:` problem naming its manifest entry and
    # bag, and the readable ones keep their Ticks in order
    good = ticks.Ticks(601, 1, {"/cmd": 0})
    calls = []

    def of_bag(bag, obs, fps, actions):
        calls.append((bag, obs, fps, actions))
        if bag == "bad_bag":
            raise RuntimeError("Failed to open bag")
        return good

    monkeypatch.setattr(ticks, "topics", lambda path: (["/cam"], ["/cmd"]))
    monkeypatch.setattr(ticks, "of_bag", of_bag)
    eps = [SimpleNamespace(bag="good_bag", where="m.json entry 0"),
           SimpleNamespace(bag="bad_bag", where="m.json entry 1")]
    got, problems = ticks.of_episodes(eps, "contract.yaml", 30)
    assert got == [good]
    assert calls == [("good_bag", ["/cam"], 30, ["/cmd"]), ("bad_bag", ["/cam"], 30, ["/cmd"])]
    (p,) = problems
    assert p.startswith("inputs: m.json entry 1:") and "bad_bag" in p and "RuntimeError" in p, p
