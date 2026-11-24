"""Golden tick streams per trigger: fires / does not fire / debounce / phase scope."""
import math

import pytest

from conftest import SPECS
from revolve_common import specs
from revolve_common.supervision import Tick, build

DONE = {"approach.done[0]": 0.2, "approach.done[1]": 0.5, "approach.done[2]": 0.5}      # none holds
ARM = {"panda_joint1": 0.0, "panda_joint2": -0.7, "panda_finger_joint1": 0.04}
BASE = (1.0, 2.0, 0.0)


@pytest.fixture(scope="module")
def task():
    return specs.resolve(SPECS, "open_fridge_door", assets=False)


def run(task, spec, ticks, **kw):
    """The Firing of one trigger over `ticks` (None if it never fires)."""
    return build([spec], task, **kw).scan(ticks)


def ticks(n, phase="approach", dt=0.1, t0=0.0, **kw):
    return [Tick(t0 + i * dt, phase, **kw) for i in range(n)]


# ------------------------------------------------------------------ verdict
def test_verdict_fires_on_a_failed_result_only(task):
    failed = {"passed": False, "failed_phase": "grasp", "reason": "timeout_s", "check": None}
    f = run(task, {"plugin": "verdict"}, [*ticks(3), Tick(0.5, "grasp", result=failed)])
    assert (f.trigger, f.reason, f.phase) == ("verdict", "grasp:timeout_s", "grasp")
    assert run(task, {"plugin": "verdict"}, [*ticks(3), Tick(0.5, None, result={"passed": True})]) is None
    assert run(task, {"plugin": "verdict"}, ticks(5)) is None                  # live: no result ever
    w = {**failed, "failed_phase": "open", "reason": "while", "check": "open.while[1]"}
    assert run(task, {"plugin": "verdict"}, [Tick(1.0, "open", result=w)]).reason == "open:while:open.while[1]"


def test_verdict_phase_scope(task):
    failed = {"passed": False, "failed_phase": "grasp", "reason": "timeout_s"}
    assert run(task, {"plugin": "verdict", "phases": ["open"]}, [Tick(1.0, "grasp", result=failed)]) is None
    assert run(task, {"plugin": "verdict", "phases": ["grasp"]}, [Tick(1.0, "grasp", result=failed)])


# ------------------------------------------------------------------ condition
SLIP = {"plugin": "condition", "phase": "open", "value": "open.while[1]", "op": "gt", "threshold": 0.01, "for_s": 0.2}


def slip(*vals, dt=0.1):
    return [Tick(i * dt, "open", {"open.while[1]": v}) for i, v in enumerate(vals)]


def test_condition_fires_after_for_s(task):
    f = run(task, SLIP, slip(0.0, 0.02, 0.02, 0.02, 0.02))
    assert f.t == pytest.approx(0.3) and f.reason == "open.while[1]=0.02 gt 0.01 for 0.2 s"


def test_condition_debounce_restarts_when_the_condition_drops(task):
    assert run(task, SLIP, slip(0.02, 0.02, 0.0, 0.02, 0.02, 0.0, 0.02)) is None


def test_condition_none_value_never_holds(task):
    assert run(task, SLIP, slip(None, None, None, None)) is None
    assert run(task, SLIP, [Tick(i * 0.1, "open", {}) for i in range(5)]) is None


def test_condition_phase_scope_and_immediate_fire(task):
    spec = {**SLIP, "for_s": 0}
    assert run(task, spec, [Tick(0.0, "grasp", {"open.while[1]": 1.0})]) is None
    assert run(task, spec, slip(1.0)).t == 0.0


@pytest.mark.parametrize("op, v, fires", [("lt", 0.9, True), ("lt", 1.0, False), ("le", 1.0, True), ("gt", 1.0, False),
                                          ("ge", 1.0, True), ("eq", 1.0, True), ("ne", 1.0, False)])
def test_condition_ops(task, op, v, fires):
    spec = {"plugin": "condition", "value": "sanity.penetration", "op": op, "threshold": 1.0}
    assert bool(run(task, spec, [Tick(0.0, "approach", {"sanity.penetration": v})])) is fires


# ------------------------------------------------------------------ phase_time
def test_phase_time_fraction_of_the_phase_timeout(task):
    spec = {"plugin": "phase_time", "phase": "approach", "fraction": 0.7}       # timeout 20 s -> 14 s
    f = run(task, spec, ticks(200, dt=0.1))
    assert f.t == pytest.approx(14.0) and f.phase == "approach"
    assert run(task, spec, ticks(139, dt=0.1)) is None


def test_phase_time_clock_restarts_per_phase(task):
    spec = {"plugin": "phase_time", "fraction": 0.5}                            # approach 10 s, grasp 2.5 s, open 7.5 s
    stream = [*ticks(99, "approach"), *ticks(24, "grasp", t0=9.9), *ticks(5, "open", t0=12.3)]
    assert run(task, spec, stream) is None
    f = run(task, spec, [*ticks(99, "approach"), *ticks(30, "grasp", t0=9.9)])
    assert f.phase == "grasp" and f.t == pytest.approx(9.9 + 2.5)


def test_phase_time_after_s(task):
    f = run(task, {"plugin": "phase_time", "after_s": 1.0}, ticks(30))
    assert f.t == pytest.approx(1.0)


def test_phase_time_scope(task):
    spec = {"plugin": "phase_time", "phases": ["open"], "after_s": 0.5}
    assert run(task, spec, ticks(30, "approach")) is None
    assert run(task, spec, ticks(30, "open")).phase == "open"


def test_phase_time_ignores_global_failure_tick(task):
    assert run(task, {"plugin": "phase_time", "fraction": 0.1}, [Tick(0.0, "global"), Tick(50.0, "global")]) is None


# ------------------------------------------------------------------ stall
STALL = {"plugin": "stall", "window_s": 1.0, "min_motion": 0.01}


def still(n, **kw):
    return ticks(n, values=DONE, joints=ARM, base=BASE, **kw)


def test_stall_fires_when_nothing_moves_for_the_window(task):
    f = run(task, STALL, still(30))
    assert f.t == pytest.approx(1.0) and f.reason.startswith("stalled: moved 0.0000 < 0.01 in 1 s")


def test_stall_needs_a_full_window(task):
    assert run(task, STALL, still(10)) is None                                  # 0.9 s


def test_stall_not_when_the_arm_moves_but_the_window_slides(task):
    moving = [Tick(i * 0.1, "approach", DONE, {**ARM, "panda_joint1": 0.05 * i}, BASE) for i in range(20)]
    assert run(task, STALL, moving) is None
    # moves for 1 s then stops: fires once the window has slid past the motion
    mixed = [Tick(i * 0.1, "approach", DONE, {**ARM, "panda_joint1": 0.05 * min(i, 10)}, BASE) for i in range(40)]
    f = run(task, STALL, mixed)
    assert f.t == pytest.approx(2.0)


def test_stall_ignores_the_gripper_but_not_the_base(task):
    grip = [Tick(i * 0.1, "approach", DONE, {**ARM, "panda_finger_joint1": 0.04 - 0.003 * i}, BASE) for i in range(20)]
    assert run(task, STALL, grip) is not None
    drive = [Tick(i * 0.1, "approach", DONE, ARM, (1.0 + 0.02 * i, 2.0, 0.0)) for i in range(20)]
    assert run(task, STALL, drive) is None
    assert run(task, {**STALL, "include_base": False}, drive) is not None


def test_stall_base_yaw_wraps(task):
    yaw = [math.pi - 0.004 + 0.0004 * i for i in range(20)]                     # crosses +pi: 0.0076 rad of motion
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi                    # noqa: E731
    stream = [Tick(i * 0.1, "approach", DONE, ARM, (1.0, 2.0, wrap(a))) for i, a in enumerate(yaw)]
    assert max(t.base[2] for t in stream) - min(t.base[2] for t in stream) > 6
    assert run(task, STALL, stream) is not None


def test_stall_not_when_the_phase_is_done(task):
    holds = {"approach.done[0]": 0.01, "approach.done[1]": 0.1, "approach.done[2]": 0.1}
    assert run(task, STALL, ticks(30, values=holds, joints=ARM, base=BASE)) is None


def test_stall_joint_selection(task):
    only = {"plugin": "stall", "window_s": 1.0, "min_motion": 0.01, "joints": ["panda_joint2"], "include_base": False}
    j1 = [Tick(i * 0.1, "approach", DONE, {**ARM, "panda_joint1": 0.1 * i}) for i in range(20)]
    assert run(task, only, j1) is not None                                      # joint1 is not watched
    assert run(task, {**STALL, "joints": "all", "include_base": False}, [
        Tick(i * 0.1, "approach", DONE, {**ARM, "panda_finger_joint1": 0.01 * i}) for i in range(20)]) is None
    arm = [Tick(i * 0.1, "approach", DONE, {"a": 0.0, "b": 0.1 * i}) for i in range(20)]
    assert run(task, {**STALL, "include_base": False}, arm) is None
    assert run(task, {**STALL, "include_base": False}, arm, arm_joints=["a"]) is not None


def test_stall_without_data_never_fires(task):
    assert run(task, STALL, ticks(50, values=DONE)) is None                     # no joints, no base
    assert run(task, STALL, ticks(50, values=DONE, joints=ARM)) is None         # base wanted but missing
    assert run(task, {**STALL, "include_base": False}, ticks(50, values=DONE, joints=ARM))


def test_stall_window_restarts_with_the_phase(task):
    stream = [*ticks(8, "approach", values=DONE, joints=ARM, base=BASE),
              *ticks(8, "grasp", t0=0.8, values=DONE, joints=ARM, base=BASE)]
    assert run(task, STALL, stream) is None
