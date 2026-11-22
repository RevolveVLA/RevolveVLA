"""success.score on synthetic state logs: a door pulled open 0 -> 65 deg in 10 s and held, with the TCP carried along the
bar (R = identity, so the bar axis is z) or not."""
import numpy as np
import pytest

from revolve_common import specs, success
from conftest import SPECS

HINGE = np.array([-1.2, 2.4, 1.75])
GRASP = np.array([[0, 0, 0, -1.79], [0, 0, 0, 2.35], [0, 0, 0, 0.97], [0, 0, 0, 1.0]])
DT, N = 0.001, 14000


@pytest.fixture(scope="module")
def rule():
    t = specs.load(SPECS / "tasks" / "open_fridge_door" / "task.yaml", robot="mobile_panda").task
    return t.success, t.boundaries.policy.grip_closed_max_m


def log(open_deg=65.0, open_s=10.0, held=True, left=True, right=True, width=0.007, grip=0.0, off_bar=0.0, open_at=1.0):
    """The door rises linearly from open_at to open_at + open_s and stays; the TCP follows the bar (off_bar m aside)."""
    t = np.arange(N) * DT
    q = np.radians(open_deg) * np.clip((t - open_at) / open_s, 0, 1)
    d = GRASP[:2, 3] - HINGE[:2]
    c, s = np.cos(q), np.sin(q)
    bar = np.stack([HINGE[0] + c * d[0] - s * d[1], HINGE[1] + s * d[0] + c * d[1]], 1)
    pos = np.column_stack([bar + off_bar, np.full(N, 1.0)]) if held else np.column_stack([np.tile(GRASP[:2, 3] + 0.5, (N, 1)), np.full(N, 1.0)])
    tcp = np.column_stack([pos, np.tile([1.0, 0, 0, 0], (N, 1))])
    return dict(door_pos=q, action_grip=np.full(N, grip), tcp_pose=tcp, handle_in_tcp=HINGE - pos, control_dt=np.array(DT),
                robot_joint_names=np.array(["base_x", "panda_finger_joint_left", "panda_finger_joint_right"]),
                robot_joint_pos=np.column_stack([np.zeros(N), np.full(N, width), np.full(N, width)]),
                contact_left_finger_handle=np.full(N, left), contact_right_finger_handle=np.full(N, right))


def score(rule, **kw):
    return success.score(log(**kw), GRASP, *rule)


def test_a_held_opening_passes(rule):
    r = score(rule)
    assert r["passed"] and r["opened"] and r["held_fraction"] == 1.0 and r["door_max_deg"] == 65.0


def test_a_push_opens_without_passing(rule):
    r = score(rule, held=False)
    assert r["opened"] and not r["passed"] and r["held_fraction"] == 0.0
    assert not score(rule, open_deg=50.0)["opened"]


@pytest.mark.parametrize("kw, why", [
    (dict(open_deg=50.0), "door never reaches 60"),
    (dict(left=False), "one finger on the handle"),
    (dict(right=False), "one finger on the handle"),
    (dict(held=False), "bar far from the TCP"),
    (dict(off_bar=0.05), "bar outside the finger gap"),
    (dict(grip=0.01), "gripper not closed"),
    (dict(width=0.001), "fingers closed on nothing"),
    (dict(width=0.02), "fingers wide open"),
    (dict(open_deg=60.4, open_s=10.0, open_at=3.9), "60 deg only reached for under 1 s"),
])
def test_failures(rule, kw, why):
    r = score(rule, **kw)
    assert not r["passed"], why


def test_door_must_stay_open_for_hold_s(rule):
    """Reaches 60 deg at 5.6 s; shut again after 0.5 s: fails; after 1.1 s: passes."""
    for end, ok in ((6100, False), (6700, True)):
        l = log(open_deg=65.0, open_s=5.0, open_at=1.0)
        l["door_pos"][end:] = 0.0
        assert success.score(l, GRASP, *rule)["passed"] is ok, end


def test_held_fraction_threshold(rule):
    """Only the first half of the travel is held (the TCP leaves the bar, then catches up): 0.5 < 0.9."""
    l = log()
    pos = l["tcp_pose"][:, :3].copy()
    pos[N // 2 - 4000:N // 2 + 2000, :2] += 0.5    # off the bar around mid-travel
    l["tcp_pose"][:, :3] = pos
    l["handle_in_tcp"] = HINGE - pos
    r = success.score(l, GRASP, *rule)
    assert 0.0 < r["held_fraction"] < 0.9 and not r["passed"]


def test_helpers():
    assert success.within_last(np.array([0, 1, 0, 0, 0, 0], bool), 3).tolist() == [False, True, True, True, False, False]
    assert success.run_length(np.array([1, 1, 0, 1, 1, 1], bool)).tolist() == [1, 2, 0, 1, 2, 3]
    r = success.rotations(np.array([[0, 0, 0, np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]]))[0]    # +90 deg about z
    assert np.allclose(r @ [1, 0, 0], [0, 1, 0])
