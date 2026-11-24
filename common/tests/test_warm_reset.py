"""warm_reset without ROS: a simulated robot (kinematic base, linear arm) stands in for the sim's topics."""
import math

import pytest

from revolve_common import sim_session as ss
from revolve_common import warm_reset as W
from revolve_common.sim_client import SimClient, WarmResetMissed, run_client
from revolve_common.sim_fake import FakeClient, FakeSession

JOINTS = ("j1", "j2")


def target(base=(0.1, -0.05, 0.2), arm=(0.3, -0.5)):
    return W.ResetTarget("mobile_panda", base, arm, JOINTS, "arm_controller")


class SimPlant:
    """The robot after a reset at `start`: the base integrates the last twist (body frame, held until replaced), the arm goes
    linearly to the last command. `stuck`: the base ignores twists. Time only moves in `sleep`."""

    def __init__(self, start=(0.0, 0.0, 0.0), arm=(0.0, 0.0), stuck=False, silent=False):
        self.t, self.x, self.v, self.q, self.stuck, self.silent = 0.0, list(start), (0, 0, 0), list(arm), stuck, silent
        self.goal, self.sent, self.forgot = None, [], 0

    def forget(self):
        self.forgot += 1

    def pose(self):
        return None if self.silent else tuple(self.x)

    def joints(self):
        return None if self.silent else dict(zip(JOINTS, self.q))

    def twist(self, v):
        self.v = v
        self.sent.append((self.t, tuple(v)))

    def command_arm(self, q, duration_s):
        self.goal = (list(q), duration_s)

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        if not self.stuck:
            c, s = math.cos(self.x[2]), math.sin(self.x[2])
            self.x[0] += dt * (c * self.v[0] - s * self.v[1])
            self.x[1] += dt * (s * self.v[0] + c * self.v[1])
            self.x[2] += dt * self.v[2]
        if self.goal:
            q, dur = self.goal
            self.q = [a + (b - a) * min(1.0, dt / max(dur, dt)) for a, b in zip(self.q, q)]
            self.goal = (q, max(dur - dt, 1e-6))


def test_it_reaches_the_target_within_the_tolerance_and_stops():
    p = SimPlant()
    r = W.run(p, target())
    assert r.ok and r.reason is None and p.forgot == 1
    assert r.base_err_m <= 0.03 and r.yaw_err_rad <= 0.03 and r.arm_err_rad <= 0.03 and 2.0 <= r.sim_s < 15
    assert p.sent[-1][1] == (0.0, 0.0, 0.0) and p.goal[1] >= 0.0


def test_it_reaches_targets_in_every_direction_with_a_turned_start():
    for start, base in [((1.0, 2.0, 1.5), (1.2, 1.9, 1.3)), ((0.0, 0.0, -3.0), (-0.1, 0.2, 3.0)), ((0, 0, 0), (0, 0, 0.5))]:
        r = W.run(SimPlant(start), target(base=base))
        assert r.ok, (start, base, r)


def test_the_twist_is_ramped_clipped_and_in_the_body_frame():
    t, tol = target(base=(0.0, 1.0, 0.0)), W.Tolerance()
    assert W.base_command((0, 0, 0), t, tol, 0.0) == (0.0, 0.0, 0.0)                            # ramp starts at zero
    vx, vy, w = W.base_command((0, 0, 0), t, tol, 10.0)
    assert (vx, vy, w) == pytest.approx((0.0, 0.2, 0.0)) and W.base_command((0, 0, 0), t, tol, 0.25)[1] == pytest.approx(0.1)
    vx, vy, _ = W.base_command((0, 0, -math.pi / 2), t, tol, 10.0)                               # facing -y: +y is behind it
    assert (vx, vy) == pytest.approx((-0.2, 0.0), abs=1e-9)
    assert W.base_command((0, 0.99, 0), t, tol, 10.0) == (0.0, 0.0, 0.0)                         # inside half the tolerance
    assert W.base_command((0, 0, 0), target(base=(0, 0, 3.0)), tol, 10.0)[2] == pytest.approx(0.4)   # clipped; ccw
    assert W.base_command((0, 0, 3.0), target(base=(0, 0, -3.0)), tol, 10.0)[2] > 0              # the short way round


def test_the_body_frame_direction():
    t = target(base=(1.0, 0.0, 0.0))
    vx, vy, _ = W.base_command((0, 0, math.pi / 2), t, W.Tolerance(), 10.0)                      # facing +y: the goal is on its right
    assert vx == pytest.approx(0.0, abs=1e-9) and vy == pytest.approx(-0.2)


@pytest.mark.parametrize("start,arm,what", [((0.5, 0, 0), (0, 0), "base delta"), ((0, 0, 1.0), (0, 0), "yaw delta"),
                                            ((0, 0, 0), (2.0, 0), "arm delta")])
def test_too_large_deltas_are_refused_with_a_reason_and_nothing_is_commanded(start, arm, what):
    p = SimPlant(start, arm)
    r = W.run(p, target(base=(0, 0, 0), arm=(0, 0)))
    assert not r.ok and what in r.reason and p.sent == [] and p.goal is None and r.base_err_m is not None


def test_a_robot_that_does_not_move_misses_with_the_errors_in_the_reason():
    r = W.run(SimPlant(stuck=True), target(), timeout_s=3.0)
    assert not r.ok and "missed the target" in r.reason and r.base_err_m > 0.05


def test_no_pose_after_the_reset_is_a_miss():
    r = W.run(SimPlant(silent=True), target())
    assert not r.ok and "no robot pose" in r.reason


def test_the_target_round_trips_through_json_and_a_report_has_times():
    t = target()
    assert W.ResetTarget.from_json(t.to_json()) == t
    r = W.ResetReport(True, 0.01).with_times(1.23456, 2.0)
    assert (r.wall_s, r.sim_s) == (1.235, 2.0) and r.to_dict()["ok"] is True


# ------------------------------------------------------------------ SimClient.reset_to
class Client(SimClient):
    def __init__(self, plant):
        super().__init__(out=None)
        self.plant, self.calls = plant, []

    def reset(self, timeout_s=0):
        self.calls.append("reset")
        self.plant.x = [0.0, 0.0, 0.0]

    def _plant(self, target):
        self.calls.append("plant")
        return self.plant


def test_reset_to_resets_first_then_drives_and_reports_wall_and_sim_time():
    c = Client(SimPlant((9.0, 9.0, 0.0)))                      # wherever it was: the reset brings it to the start
    r = c.reset_to(target())
    assert c.calls == ["reset", "plant"] and r.ok and r.wall_s >= 0 and r.sim_s > 0


def test_a_miss_is_a_report_and_run_client_turns_the_exception_into_the_exit_code(monkeypatch):
    r = Client(SimPlant(stuck=True)).reset_to(target(), timeout_s=2.0)
    assert not r.ok
    codes = []
    monkeypatch.setattr(SimClient, "connect", classmethod(lambda cls, out: object()))
    monkeypatch.setattr("os._exit", codes.append)

    def miss(client):
        raise WarmResetMissed(r)
    run_client(miss, "out")
    assert codes == [W.MISS_EXIT] and W.MISS_EXIT not in ss.EXIT_CODES.values()


def test_the_fake_client_reaches_or_misses_on_demand(tmp_path):
    s = FakeSession(tmp_path / "s.yaml", "mobile_panda", "open_fridge_door", tmp_path / "sim", "n", 100,
                    reaches=lambda t: t.base[0] < 1)
    s.start()
    c = FakeClient(s)
    assert c.reset_to(target()).ok and c.resets() == 1
    miss = c.reset_to(target(base=(2, 0, 0)))
    assert not miss.ok and "missed" in miss.reason and s.warm_targets[-1].base[0] == 2 and c.resets() == 2


# ------------------------------------------------------------------ world_key / same_world
def scene(tmp_path, name, start, x=1.0, extra=None):
    d = tmp_path / name
    d.mkdir()
    (d / "scene.yaml").write_text(f"name: kitchen\nassets: [a]\nmodels:\n  fridge: {{model: fridge, pose: [{x}, 0, 0, 0]}}\n"
                                  f"robot_start: {{base: {start}, arm: folded}}\n{extra or ''}")
    return d / "scene.yaml"


def test_same_world_ignores_only_the_robot_start(tmp_path):
    a, b = scene(tmp_path, "a", "[0, 0, 0]"), scene(tmp_path, "b", "[0.2, -0.1, 0.3]")
    assert ss.same_world(a, b) and ss.world_key(a) == ss.world_key(b)
    assert not ss.same_world(a, scene(tmp_path, "c", "[0, 0, 0]", x=2.0))                       # the fridge moved
    assert not ss.same_world(a, scene(tmp_path, "d", "[0, 0, 0]", extra="extra: 1\n"))
    assert ss.world_key(tmp_path / "missing.yaml") is None and not ss.same_world(a, tmp_path / "missing.yaml")
    assert not ss.same_world(tmp_path / "missing.yaml", tmp_path / "missing.yaml")
