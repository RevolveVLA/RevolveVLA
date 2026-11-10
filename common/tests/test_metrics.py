import math

import numpy as np
import pytest

from revolve_common import metrics as M
from revolve_common.specs import Condition


def C(**kw):
    return Condition(**kw)


def pose(p, yaw_deg=0.0):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = p
    return T


def test_tcp_errors():
    s = dict(tcp_pose=pose([1.0, 0.0, 1.0], 5.0), grasp_pose=pose([1.0, 0.012, 1.0]))
    assert M.tcp_position_error_m(s) == pytest.approx(0.012)
    assert M.tcp_rotation_error_deg(s) == pytest.approx(5.0)
    assert M.evaluate(C(metric="tcp_position_error_m", le=0.015), s)["pass_"]
    assert not M.evaluate(C(metric="tcp_position_error_m", le=0.010), s)["pass_"]


def test_tcp_error_to_grasp_segment():
    s = dict(tcp_pose=pose([1.0, 0.003, 1.05]), grasp_pose=pose([1.0, 0.0, 0.965]),
             grasp_segment=[[1.0, 0.0, 0.965], [1.0, 0.0, 1.085]])
    assert M.tcp_position_error_m(s) == pytest.approx(0.003)


def test_grasp_width_live_and_offline():
    s = dict(fingers=[0.0137, 0.0003], grasp_half_width_m=0.007)
    c = C(metric="grasp_width_error_m", le=0.008, offline=0.006)
    assert M.grasp_width_error_m(s) == pytest.approx(0.0067)
    assert M.evaluate(c, s)["pass_"]
    r = M.evaluate(c, s, offline=True)
    assert not r["pass_"] and r["threshold"] == 0.006


def test_finger_contacts():
    s = dict(contact_left_finger_handle=[False, True, False], contact_right_finger_handle=[False, False])
    assert M.finger_contacts(s) == 1
    assert not M.evaluate(C(metric="finger_contacts", eq=2), s)["pass_"]
    s["contact_right_finger_handle"] = [True]
    assert M.evaluate(C(metric="finger_contacts", eq=2), s)["pass_"]


def test_handle_slip_uses_last_row():
    s = dict(handle_in_tcp=[[0, 0, 0], [0, 0.003, 0.004]], handle_in_tcp_ref=[0, 0, 0])
    assert M.handle_slip_m(s) == pytest.approx(0.005)


def series(values, dt=0.1, key="top_door_hinge", conv=math.radians):
    return [dict(t=i * dt, joints={key: conv(v)}) for i, v in enumerate(values)]


def test_hold_s_every_sample_of_the_window():
    c = C(metric="joint_deg", joint="top_door_hinge", ge=60.0, hold_s=1.0)
    # 0.0 .. 2.0 s; the last 1.0 s (t >= 1.0) is all >= 60
    ok = series([0, 20, 40, 55, 59, 61, 62, 63, 64, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66])
    r = M.evaluate(c, ok)
    assert r["pass_"] and r["value"] == pytest.approx(66.0)
    # one dip to 59.9 at t 1.5 inside the hold
    bad = [dict(s) for s in ok]
    bad[15] = dict(t=1.5, joints={"top_door_hinge": math.radians(59.9)})
    r = M.evaluate(c, bad)
    assert not r["pass_"] and r["value"] == pytest.approx(59.9)
    # a dip just before the window does not count
    early = [dict(s) for s in ok]
    early[9] = dict(t=0.9, joints={"top_door_hinge": math.radians(10)})
    assert M.evaluate(c, early)["pass_"]


def test_hold_s_series_shorter_than_hold_fails():
    c = C(metric="joint_cm", joint="middle_slide_top", ge=15.0, hold_s=1.0)
    s = series([20, 20, 20], dt=0.1, key="middle_slide_top", conv=lambda v: v / 100)
    r = M.evaluate(c, s)
    assert not r["pass_"] and "hold window" in r["note"]


def test_targets_static_matches_hold_duration():
    t = np.zeros((2000, 10))
    t[:1200, 0] = np.linspace(0, 1, 1200)          # moves until step 1199, static after
    t[1200:, 0] = 1.0
    s = dict(targets=t, dt=0.001, change_tol=1e-9)
    assert M.targets_static_s(s) == pytest.approx((2000 - 1200) * 0.001)
    assert not M.evaluate(C(metric="targets_static_s", ge=1.0), s)["pass_"]
    assert M.evaluate(C(metric="targets_static_s", ge=1.0, only="plan"), s, policy=True) is None


def test_only_offline_skipped_live():
    c = C(metric="spawn_clearance_m", ge=0.0, only="offline")
    assert M.evaluate(c, dict(spawn_clearance_m=-0.001)) is None
    assert not M.evaluate(c, dict(spawn_clearance_m=-0.001), offline=True)["pass_"]


def test_sanity_metrics():
    s = dict(link_speeds=[0.1, 4.9], contact_depths=[0.002, 0.011], other_robot_contacts=[0, 1, 0],
             commands=[0.0, float("nan"), 1.0], gz_log_problems=[])
    ok, res = M.evaluate_all([C(metric="peak_body_speed_mps", lt=5.0), C(metric="penetration_m", lt=0.01),
                              C(metric="other_robot_contacts", eq=0), C(metric="non_finite_commands", eq=0),
                              C(metric="gz_log_problem_lines", eq=0)], s)
    assert not ok
    assert [r["pass_"] for r in res] == [True, False, False, False, True]
    assert M.peak_body_speed_mps(dict(link_speeds=[1.0, float("nan")])) == math.inf


def test_task_phase_on_hand_made_state():
    """The fridge open phase on a hand-made end state, live and offline."""
    from revolve_common import specs
    from conftest import SPECS
    task = specs.load(SPECS / "tasks" / "open_fridge_door" / "task.yaml", robot="mobile_panda").task
    conds = task.phase("open").pass_.all
    states = [dict(t=round(i * 0.05, 3), joints={"top_door_hinge": math.radians(65)},
                   handle_in_tcp=[0, 0.011, 0], handle_in_tcp_ref=[0, 0, 0],
                   contact_left_finger_handle=[True], contact_right_finger_handle=[True],
                   targets=np.zeros((1500, 10)), dt=0.001, change_tol=task.boundaries.change_tol)
              for i in range(41)]
    ok, res = M.evaluate_all(conds, states)
    assert ok, res
    ok, res = M.evaluate_all(conds, states, offline=True)       # slip 11 mm > the offline 10 mm
    assert not ok and [r["metric"] for r in res if not r["pass_"]] == ["handle_slip_m"]
    ok, res = M.evaluate_all(conds, states, policy=True)
    assert ok and "targets_static_s" not in [r["metric"] for r in res]
