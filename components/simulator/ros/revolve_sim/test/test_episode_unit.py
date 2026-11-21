"""Evaluator logic without a sim: the phase rules (revolve_sim.episode) on synthetic metric streams, and the metric
geometry / contact matching (revolve_sim.metrics)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revolve_common import specs  # noqa: E402

from revolve_sim.episode import Episode  # noqa: E402
from revolve_sim.metrics import (Metrics, Ref, Snapshot, point_distance, robot_link_of, rpy_matrix,  # noqa: E402
                                 split_collision)

DT = 0.01
ROOT = Path(__file__).resolve().parents[5] / "specs"


def task(phases, budget_s=45.0, **kw):
    return specs.Task.model_validate({"name": "t", "scene": "s", "prompt": "p", "phases": phases,
                                      "budget_s": budget_s, **kw})


def sanity(**checks):
    return {k: specs.SanityCheck.model_validate(v) for k, v in checks.items()}


TWO = [{"name": "a", "timeout_s": 2.0, "done": [{"metric": "joint", "of": "m::j", "ge": 1.0}]},
       {"name": "b", "timeout_s": 3.0, "done": [{"metric": "joint", "of": "m::j", "ge": 2.0, "for_s": 0.5}],
        "while": [{"metric": "contacts", "from": "robot", "to": "m::l", "eq": 1}]}]


def run(ep, stream, t_end=60.0):
    """stream(t) -> {key: value}; step until the episode ends -> its result."""
    t = ep.t0
    while t < ep.t0 + t_end:
        t = round(t + DT, 6)
        r = ep.step(t, stream(t - ep.t0))
        if r is not None:
            return r
    raise AssertionError("episode did not end")


def test_done_advances_and_passes():
    ep = Episode(task(TWO), {}, 10.0)
    r = run(ep, lambda t: {"a.done[0]": 1.0 if t > 0.5 else 0.0, "b.done[0]": 2.5 if t > 1.0 else 0.0,
                           "b.while[0]": 1.0})
    assert r["passed"] and r["failed_phase"] is None and r["t_fail"] is None
    a, b = r["phases"]
    assert a["status"] == "passed" and a["t_end"] == pytest.approx(0.51, abs=1e-6)
    assert b["status"] == "passed" and b["t_start"] == a["t_end"]
    assert b["t_end"] == pytest.approx(1.51, abs=1e-6)          # true from 1.01 s, held 0.5 s
    assert b["last"] == {"b.done[0]": 2.5, "b.while[0]": 1.0}


def test_for_s_resets_when_false():
    ep = Episode(task(TWO), {}, 0.0)
    # phase b: true 0.4 s, false once, then true for good at 1.0 s -> done at 1.5 s
    r = run(ep, lambda t: {"a.done[0]": 1.0, "b.done[0]": 0.0 if 0.5 < t < 1.0 else 2.0, "b.while[0]": 1.0})
    assert r["passed"]
    assert r["phases"][1]["t_end"] == pytest.approx(1.5, abs=0.011)


def test_while_violation_fails_the_phase():
    ep = Episode(task(TWO), {}, 0.0)
    r = run(ep, lambda t: {"a.done[0]": 1.0, "b.done[0]": 0.0, "b.while[0]": 0.0 if t > 0.8 else 1.0})
    assert not r["passed"] and r["failed_phase"] == "b" and r["reason"] == "while"
    assert r["check"] == "b.while[0]" and r["value"] == 0.0 and r["t_fail"] == pytest.approx(0.81, abs=1e-6)
    assert [p["status"] for p in r["phases"]] == ["passed", "failed"]


def test_timeout():
    ep = Episode(task(TWO), {}, 5.0)
    r = run(ep, lambda t: {"a.done[0]": 0.0})
    assert r["failed_phase"] == "a" and r["reason"] == "timeout_s"
    assert r["t_fail"] == pytest.approx(2.01, abs=1e-6)
    assert [p["status"] for p in r["phases"]] == ["failed", "not_reached"]


def test_timeout_counts_from_the_phase_start():
    ep = Episode(task(TWO), {}, 0.0)
    r = run(ep, lambda t: {"a.done[0]": 1.0 if t > 1.5 else 0.0, "b.done[0]": 0.0, "b.while[0]": 1.0})
    assert r["failed_phase"] == "b" and r["reason"] == "timeout_s"
    assert r["t_fail"] == pytest.approx(1.51 + 3.0 + 0.01, abs=0.011)


def test_budget():
    phases = [{"name": "a", "done": [{"metric": "joint", "of": "m::j", "ge": 1.0}]}]   # no timeout_s
    r = run(Episode(task(phases, budget_s=4.0), {}, 0.0), lambda t: {"a.done[0]": 0.0})
    assert r["failed_phase"] == "a" and r["reason"] == "budget_s" and r["t_fail"] == pytest.approx(4.01, abs=1e-6)


def test_unknown_values_are_neither_true_nor_violations():
    ep = Episode(task(TWO), sanity(pen={"metric": "penetration", "lt": 0.01}), 0.0)
    r = run(ep, lambda t: {"a.done[0]": 1.0, "b.done[0]": math.nan, "b.while[0]": None, "sanity.pen": math.nan})
    assert r["failed_phase"] == "b" and r["reason"] == "timeout_s"


def test_timed_sanity_fails_the_active_phase():
    ep = Episode(task(TWO), sanity(pen={"metric": "penetration", "lt": 0.01}), 0.0)
    r = run(ep, lambda t: {"a.done[0]": 1.0 if t > 0.3 else 0.0, "b.done[0]": 0.0, "b.while[0]": 1.0,
                           "sanity.pen": 0.02 if t > 0.6 else 0.0})
    assert r["failed_phase"] == "b" and r["reason"] == "sanity" and r["check"] == "sanity.pen"
    assert r["value"] == 0.02 and r["condition"] == "penetration of=robot lt 0.01"
    assert r["sanity_last"] == {"sanity.pen": 0.02}


def test_untimed_sanity_fails_global_and_wins():
    checks = sanity(pen={"metric": "penetration", "lt": 0.01},
                    log={"metric": "gz_log_problem_lines", "eq": 0, "timed": False})
    ep = Episode(task(TWO), checks, 0.0)
    r = run(ep, lambda t: {"a.done[0]": 0.0, "sanity.pen": 0.02 if t > 0.5 else 0.0,
                           "sanity.log": 1.0 if t > 0.5 else 0.0})
    assert r["failed_phase"] == specs.GLOBAL_PHASE and r["check"] == "sanity.log"
    assert r["phases"][0]["status"] == "aborted"


def test_sanity_for_s_debounces():
    ep = Episode(task(TWO), sanity(pen={"metric": "penetration", "lt": 0.01, "for_s": 0.2}), 0.0)
    # a 0.1 s spike is tolerated, a lasting one is not
    r = run(ep, lambda t: {"a.done[0]": 0.0, "sanity.pen": 0.02 if 0.3 < t < 0.4 or t > 1.0 else 0.0})
    assert r["reason"] == "sanity" and r["t_fail"] == pytest.approx(1.21, abs=0.011)


def test_sanity_override_and_removal():
    """The merged sanity checks (tasks/sanity.yaml <- the task's overrides) drive the episode."""
    base = specs.load_sanity(ROOT)
    merged = specs.merge_sanity(base, {"penetration": specs.SanityCheck.model_validate(
        {"metric": "penetration", "lt": 0.05}), "robot_contacts": None})
    assert "robot_contacts" not in merged and merged["penetration"].value == 0.05
    ep = Episode(task(TWO), merged, 0.0)
    keys = {k for k, _ in ep.keys()}
    assert "sanity.robot_contacts" not in keys and {"sanity.penetration", "sanity.gz_log"} <= keys
    ok = {k: 0.0 for k in keys} | {"sanity.penetration": 0.03, "a.done[0]": 0.0}     # 3 cm < the new 5 cm
    r = run(ep, lambda t: ok | ({"sanity.penetration": 0.06} if t > 1.0 else {}))
    assert r["reason"] == "sanity" and r["check"] == "sanity.penetration" and r["failed_phase"] == "a"


# ------------------------------------------------------------------ metrics
def test_point_box_distance():
    box = ("box", {"size": [0.012, 0.014, 0.25]})
    assert point_distance(np.array([0.0, 0.0, 0.0]), box) == 0.0
    assert point_distance(np.array([0.106, 0.0, 0.0]), box) == pytest.approx(0.1)
    assert point_distance(np.array([0.106, 0.107, 0.0]), box) == pytest.approx(math.hypot(0.1, 0.1))
    assert point_distance(np.array([0.0, 0.0, 0.225]), box) == pytest.approx(0.1)
    assert point_distance(np.array([0.0, 0.0, 0.3]), ("sphere", {"radius": 0.1})) == pytest.approx(0.2)
    assert point_distance(np.array([0.3, 0.0, 0.0]), ("cylinder", {"radius": 0.1, "length": 1.0})) == \
        pytest.approx(0.2)


def test_rpy_matrix():
    T = rpy_matrix(1, 2, 3, 0, 0, math.pi / 2)
    assert np.allclose(T[:3, 0], [0, 1, 0]) and np.allclose(T[:3, 3], [1, 2, 3])
    T = rpy_matrix(0, 0, 0, math.pi / 2, 0, 0)
    assert np.allclose(T[:3, 2], [0, -1, 0])


def test_collision_names():
    assert split_collision("fridge::top_door::top_door_handle") == ("fridge", "top_door", "top_door_handle")
    assert robot_link_of("base_footprint", "base_footprint_fixed_joint_lump__panda_link0_collision_1") == "panda_link0"
    assert robot_link_of("panda_link7", "panda_link7_collision") == "panda_link7"
    any_ = Ref("any")
    assert any_.matches("fridge::body::c", "bot") and not any_.matches("ground_plane::link::collision", "bot")
    assert not any_.matches("bot::base_footprint::x", "bot")


class _Res:
    """A minimal specs.Resolved for Metrics."""

    def __init__(self, task_, models):
        self.task, self.models = task_, models
        self.robot = specs.Robot(name="bot", description="d", controllers="c", poses={"p": [0.0]},
                                 frames={"tcp": "tcp_link"}, groups={"fingers": ["fl", "fr"]})


def metrics():
    t = task([{"name": "a", "done": [{"metric": "joint", "of": "m::j", "ge": 1.0}]}],
             targets={"handle": "m::door::handle"})
    models = {"m": specs.ModelInfo(links={"door": {"handle", "panel"}}, joints={"j": ("revolute", 0.0, 2.0)})}
    box = (rpy_matrix(0.1, 0, 0, 0, 0, 0), ("box", {"size": [0.02, 0.02, 0.2]}))
    return Metrics(_Res(t, models), ["base", "tcp_link", "fl", "fr"], {"m": {("door", "handle"): box,
                                                                             ("door", "panel"): box}})


def cond(**kw):
    return specs.Condition.model_validate(kw)


def test_metrics_distance_angle_contacts():
    m = metrics()
    door = rpy_matrix(1.0, 0.0, 0.0, 0, 0, 0)
    tcp = rpy_matrix(1.3, 0.0, 0.0, 0, math.pi / 2, 0)          # tcp z along world +x
    snap = Snapshot(1.0, poses={("m", "door"): door, ("bot", "tcp_link"): tcp,
                                ("bot", "fl"): rpy_matrix(0, 0.04, 0, 0, 0, 0), ("bot", "fr"): rpy_matrix(0, -0.04, 0, 0, 0, 0)},
                    joints={("m", "j"): 0.5},
                    contacts=[("bot::fl::fl_collision", "m::door::handle", 0.001),
                              ("bot::fr::fr_collision", "m::door::panel", 0.004),
                              ("bot::base::base_collision", "ground_plane::link::collision", 0.002)])
    assert m.value(cond(metric="distance", **{"from": "robot::tcp"}, to="handle", le=0.1), snap) == \
        pytest.approx(0.19)                                     # 1.3 - (1.0 + 0.1 + 0.01)
    assert m.value(cond(metric="angle", **{"from": "robot::tcp"}, to="handle", axes=["z", "x"], le=0.1), snap) == \
        pytest.approx(0.0, abs=1e-9)
    assert m.value(cond(metric="angle", **{"from": "robot::tcp"}, to="handle", axes=["z", "-x"], le=0.1), snap) == \
        pytest.approx(math.pi)
    assert m.value(cond(metric="joint", of="m::j", ge=1.0), snap) == 0.5
    assert m.value(cond(metric="contacts", **{"from": "robot::fingers"}, to="handle", eq=2), snap) == 1.0
    assert m.value(cond(metric="contacts", **{"from": "robot::fingers"}, to="m", eq=2), snap) == 2.0
    assert m.value(cond(metric="contacts", **{"from": "robot"}, to="any", eq=0), snap) == 2.0
    assert m.value(cond(metric="contacts", **{"from": "robot"}, to="any_except_targets", eq=0), snap) == 1.0
    assert m.value(cond(metric="penetration", lt=0.01), snap) == 0.004
    assert m.value(cond(metric="gripper_opening", of="robot::fingers", ge=0.0), snap) == pytest.approx(0.08)


def test_metrics_slip_and_speed():
    m = metrics()
    c = cond(metric="slip", **{"from": "robot::tcp"}, to="handle", le=0.01)
    door = rpy_matrix(1.0, 0.0, 0.0, 0, 0, 0)
    for i, (dx, tdoor) in enumerate([(0.0, 0.0), (0.0, 0.5), (0.003, 0.5)]):   # the tcp follows the door, then drifts
        T = rpy_matrix(1.0 + tdoor, 0, 0, 0, 0, 0)
        snap = Snapshot(i * 0.01, poses={("m", "door"): T, ("bot", "tcp_link"): rpy_matrix(1.3 + tdoor + dx, 0, 0,
                                                                                            0, 0, 0)})
        v = m.value(c, snap, "a.while[0]")
    assert v == pytest.approx(0.003)
    m.reset_phase()
    assert m.value(c, snap, "a.while[0]") == 0.0
    del door
    s = cond(metric="peak_body_speed", lt=5.0)
    for i in range(6):                                          # tcp at 2 m/s, the rest still
        snap = Snapshot(i * 0.01, poses={("bot", "tcp_link"): rpy_matrix(0.02 * i, 0, 0, 0, 0, 0),
                                         ("bot", "base"): rpy_matrix(0, 0, 0, 0, 0, 0)})
        m.record(snap)
    assert m.value(s, snap) == pytest.approx(2.0)
    m = metrics()                                               # stale poses: differenced over their own stamps
    for i in range(6):
        ts = 0.01 * max(0, i - 3) if i < 5 else 0.05            # the tf lags, then catches up
        snap = Snapshot(i * 0.01, poses={("bot", "tcp_link"): rpy_matrix(2.0 * ts, 0, 0, 0, 0, 0)},
                        stamps={("bot", "tcp_link"): ts})
        m.record(snap)
    assert m.value(s, snap) == pytest.approx(2.0)
    m = metrics()                                               # the tf stalls for 40 ms (longer than the window)
    for i in range(9):
        ts = 0.01 * min(i, 4)
        snap = Snapshot(i * 0.01, poses={("bot", "tcp_link"): rpy_matrix(2.0 * ts, 0, 0, 0, 0, 0)},
                        stamps={("bot", "tcp_link"): ts})
        m.record(snap)
    assert m.value(s, snap) == pytest.approx(2.0)


def test_metrics_counters():
    m = metrics()
    snap = Snapshot(0.0, commands=2, gz_lines=["(t) [error] [X.cc:1] bad", "(t) [error] [Y.cc:2] known benign"])
    assert m.value(cond(metric="non_finite_commands", eq=0), snap) == 2.0
    assert m.value(cond(metric="gz_log_problem_lines", eq=0, ignore=["known benign"]), snap) == 1.0
