"""policy_io: the base twist in the robot frame from base poses (dataset, live closed loop, oracle) and the
contract layout (no world-frame base pose in observation.state or action)."""
import math

import numpy as np
import pytest

from revolve_common import policy_io as P

DT, FPS = 0.001, 30


def test_body_twist_is_in_the_robot_frame():
    # facing +y (yaw pi/2) and moving +y in the world at 0.3 m/s: straight ahead
    v = P.body_twist([1.0, 2.0, math.pi / 2], [1.0, 2.3, math.pi / 2], 1.0)
    assert np.allclose(v, [0.3, 0.0, 0.0], atol=1e-12)
    # facing +x, moving +y: to the left
    assert np.allclose(P.body_twist([0, 0, 0], [0, 0.3, 0], 1.0), [0, 0.3, 0], atol=1e-12)
    # facing -x (yaw pi, the drawer facing), moving +x in the world: backwards; any world offset or 2 pi k gives
    # the same twist
    for b0 in ([0, 0, math.pi], [-2.6, 0.3, math.pi + 2 * math.pi]):
        assert np.allclose(P.body_twist(b0, np.add(b0, [0.1, 0, 0.05]), 0.5), [-0.2, 0, 0.1], atol=6e-3)   # mid yaw
    w = P.world_velocity(math.pi / 2, [0.3, 0, 0.2])
    assert np.allclose(w, [0, 0.3, 0.2], atol=1e-12)


def drive(n=6000):
    """A smooth whole-body base path (m, rad): curve forward, turn, slide sideways, back off."""
    t = np.arange(n) * DT
    s = 0.5 - 0.5 * np.cos(np.pi * t / t[-1])
    x = -1.0 + 1.2 * s + 0.1 * np.sin(2 * np.pi * s)
    y = 0.3 * np.sin(np.pi * s) ** 2
    yaw = math.pi + 0.35 * np.sin(2 * np.pi * s)
    return np.stack([x, y, yaw], -1)


def test_measured_twist_constant_motion_and_window():
    n = 400
    yaw0 = 0.7
    ahead, w = 0.25, 0.3            # m/s ahead and rad/s: an arc
    b = np.zeros((n, 3))
    b[0] = [0.5, -0.2, yaw0]
    for k in range(1, n):
        b[k] = P.integrate(b[k - 1], [ahead, 0.0, w], DT)
    m = P.measured_twist(b, FPS, DT)
    L = P.lag_steps(FPS, DT)
    assert L == 33 and m.shape == (n, 3)
    assert np.allclose(m[L:], [ahead, 0.0, w], atol=1e-4)
    assert np.allclose(m[0], 0) and np.allclose(m[5], [ahead * 5 / L, 0, w * 5 / L], atol=1e-4)   # warm-up
    assert all(np.allclose(P.measured_twist_at(b, k, FPS, DT), m[k]) for k in (0, 10, L, n - 1))


def test_command_twist_integrates_back_to_the_path():
    """The dataset's actions, each held for one frame period and integrated every 1 ms step as closed_loop_system
    does, retrace the commanded base path (the oracle's premise)."""
    B = drive()
    log = dict(control_dt=DT, time_s=np.arange(len(B)) * DT, action_base=B, action_arm=np.zeros((len(B), 5)),
               action_grip=np.full(len(B), 0.04))
    rows, A = P.episode_actions(log, FPS)
    assert A.shape == (len(rows), 3 + 5 + 1) and rows[1] == 33 and rows[3] == 99   # tick k x int(1e9 / 30) ns
    b = B[0].copy()
    err = 0.0
    for j, k in enumerate(rows):
        k1 = rows[j + 1] if j + 1 < len(rows) else len(B)
        for _ in range(k, k1):
            b = P.integrate(b, A[j, :3], DT)
        if k1 < len(B):
            err = max(err, float(np.abs(b[:2] - B[k1, :2]).max()), float(abs(b[2] - B[k1, 2])))
    assert err < 2e-3, err                           # m / rad over a 6 s, 1.4 m drive
    # no world pose in the action: a translated, turned copy of the path gives the same twists
    c, s = math.cos(1.0), math.sin(1.0)
    B2 = np.stack([c * B[:, 0] - s * B[:, 1] + 3.0, s * B[:, 0] + c * B[:, 1] - 1.0, B[:, 2] + 1.0], -1)
    _, A2 = P.episode_actions(dict(log, action_base=B2), FPS)
    assert np.allclose(A2, A, atol=1e-9)


def test_command_twist_is_smooth_on_a_smooth_path():
    v = P.command_twist(drive(), FPS, DT)
    rough = np.abs(np.diff(v, 2, axis=0)).max(0)     # per-step second difference
    assert (rough < 1e-4).all(), rough


def test_policy_rows():
    s = P.policy_state(np.ones((4, 3)), np.zeros((4, 5)), np.full((4, 1), 0.3))
    a = P.policy_action(np.ones((4, 3)), np.zeros((4, 5)), np.full(4, 0.04))
    assert s.shape == (4, 9) and a.shape == (4, 9) and (a[:, -1] == 0.04).all()


def test_contract_layout_rejects_world_pose_and_bad_twist():
    bad = {"channel": {"topic": "/joint_states", "type": P.JS}, "select": ["velocity.base_x"]}
    with pytest.raises(ValueError, match="positions only"):
        P.sources(bad)
    with pytest.raises(ValueError, match="want"):
        P.sources({"channel": {"topic": "/cmd_vel", "type": P.TWIST}, "select": ["linear.x", "linear.y"]})
    with pytest.raises(ValueError, match="not one of"):
        P.sources({"channel": {"topic": "/p", "type": "geometry_msgs/msg/PoseStamped"}, "select": ["pose.position.x"]})


def wrapped(b):
    """The same base poses with the yaw logged wrapped to (-pi, pi], as odometry and gz poses log it."""
    b = np.array(b, float)
    b[..., 2] = np.angle(np.exp(1j * b[..., 2]))
    return b


def test_body_twist_takes_the_shortest_rotation():
    # 3.1 then 3.153 rad, logged as 3.1 then -3.13: a 0.053 rad turn, not a 6.23 rad one (188 rad/s over 1/30 s)
    b0, b1 = [0.0, 0.0, 3.1], [0.01, 0.0, 3.153]
    assert np.allclose(P.body_twist(b0, wrapped(b1), 1 / FPS), P.body_twist(b0, b1, 1 / FPS), atol=1e-9)
    # a change below pi is kept as it is, not wrapped the other way
    assert np.isclose(P.body_twist([0, 0, 0.1], [0, 0, 3.0], 1.0)[2], 2.9)


def test_twists_of_a_path_logged_wrapped():
    B = drive()                                       # its yaw swings across pi
    assert np.ptp(wrapped(B)[:, 2]) > 6               # the log jumps
    for twist in (P.measured_twist, P.command_twist):
        assert np.allclose(twist(wrapped(B), FPS, DT), twist(B, FPS, DT), atol=1e-9)
