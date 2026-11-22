"""The `sim` rollout plugin, the policy client and the oracle's actions (CPU: FakeSession + the client loop in this process; the
container, ROS and the bag reader are the GPU test's)."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from datagen.core.interfaces import Config
from datagen.core.spec import SpecError

from evaluation import oracle_actions as O
from evaluation import rollout_client as C
from evaluation.core.errors import EvalError
from evaluation.core.interfaces import EvalItem
from evaluation.core.registry import REGISTRY
from evaluation.core.sims import sim_prefix
from evaluation.core.spec import RunnerSpec
from evaluation.plugins.rollouts import sim as S
from revolve_common.sim_fake import FakeClient, FakeSession
from revolve_common.sim_session import REPO, DomainPool, SimFault

spec = SimpleNamespace(train=SimpleNamespace(task="open_fridge_door", robot="mobile_panda"), out="runs/eval/x")
smolvla, oracle = (REGISTRY.create("policy", n, spec, {}) for n in ("smolvla", "oracle"))


# ------------------------------------------------------------------ the oracle's actions
def msg(t, **kw):
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=int(t), nanosec=int(round((t % 1) * 1e9)))), **kw)


def state(t, q):
    return msg(t, reference=SimpleNamespace(positions=q), joint_names=["j1", "j2"][:len(q)])


def twist(t, x, y=0.0, w=0.0):
    return msg(t, twist=SimpleNamespace(linear=SimpleNamespace(x=x, y=y), angular=SimpleNamespace(z=w)))


def test_the_series_holds_the_last_reference_and_the_base_twist_for_its_timeout():
    ts = [10.0 + i * 0.001 for i in range(2001)]                              # 1 kHz states for 2 s
    arm = [state(t, [t - 10.0, 2 * (t - 10.0)]) for t in ts]
    hand = [state(t, [0.04 if t < 11.0 else -0.04]) for t in ts]
    cmds = [twist(10.2, 0.2), twist(10.3, 0.1, 0.05, 0.3)]                    # the last one expires 0.5 s later
    s = O.from_messages({"arm": arm, "hand": hand}, cmds)
    assert s.ticks == 60 and s.joints == {"arm": ["j1", "j2"], "hand": ["j1"]}
    assert s.positions["arm"].shape == (60, 2) and s.positions["hand"].shape == (60, 1)
    assert s.positions["arm"][30, 0] == pytest.approx(1.0, abs=1e-3)           # tick 30 = 1 s
    assert s.positions["hand"][29, 0] == 0.04 and s.positions["hand"][31, 0] == -0.04
    assert np.allclose(s.base[5], 0) and np.allclose(s.base[7], [0.2, 0, 0])   # 10.233 s: the first twist
    assert np.allclose(s.base[10], [0.1, 0.05, 0.3])                           # 10.333
    assert np.allclose(s.base[25], 0)                                          # > 0.5 s after the last: stopped
    assert np.isfinite(s.positions["arm"]).all()


def test_the_series_runs_over_the_span_all_controllers_have_and_holds_after_its_end():
    a = [state(0.0 + i * 0.01, [1.0]) for i in range(101)]                    # 0..1 s
    b = [state(0.5 + i * 0.01, [2.0]) for i in range(101)]                    # 0.5..1.5 s
    s = O.from_messages({"a": a, "b": b}, [])
    assert s.ticks == 15 and np.allclose(s.base, 0)                            # 0.5..1.0 s
    pos, base = s.at(100)
    assert pos["a"][0] == 1.0 and pos["b"][0] == 2.0 and np.allclose(base, 0)  # past the end: hold, base stops
    assert np.allclose(s.at(3)[0]["b"], 2.0)
    with pytest.raises(ValueError, match="no controller_state"):
        O.from_messages({}, [])
    with pytest.raises(ValueError, match="no action tick"):
        O.from_messages({"a": [state(0.0, [0.0])]}, [])


# ------------------------------------------------------------------ the client loop
class Clock:
    """A client whose sim time only moves when the loop sleeps; the result is out at `end_s` (None: never)."""

    def __init__(self, end_s=1.0):
        self.t, self.end_s = 5.0, end_s

    def now(self):
        return self.t

    def has_result(self, n):
        return self.end_s is not None and self.t >= 5.0 + self.end_s

    def sleep(self, dt):
        self.t += dt


def series(ticks=45):
    q = np.arange(ticks, dtype=float)[:, None]
    return O.ActionSeries({"arm": ["j1"]}, {"arm": q}, np.column_stack([q[:, 0] * 0.01, np.zeros(ticks), np.zeros(ticks)]))


def test_the_stream_publishes_each_due_tick_once_at_30_hz_and_holds_the_end():
    clock, sent = Clock(end_s=2.0), []
    n = C.stream(clock, 0, series(), lambda pos, base: sent.append((pos["arm"][0], base[0])), sleep=clock.sleep)
    assert n == len(sent) and 58 <= n <= 62                                   # 2 s at 30 Hz
    ticks = [p for p, _ in sent]
    assert ticks == sorted(ticks) and ticks[0] == 0.0 and ticks[44] == 44.0 and ticks[-1] == 44.0   # then held
    assert sent[44][1] == pytest.approx(0.44) and sent[-1][1] == 0.0          # the base stops with the series


def test_the_stream_gives_up_when_no_result_comes():
    clock = Clock(end_s=None)
    with pytest.raises(SimFault, match="result_timeout"):
        C.stream(clock, 0, series(), lambda *a: None, timeout_s=0.05, sleep=lambda dt: clock.sleep(dt) or __import__("time").sleep(0.001))


def test_play_resets_only_when_asked_and_writes_the_episode(tmp_path):
    sess = FakeSession(tmp_path / "scene.yaml", "mobile_panda", "open_fridge_door", tmp_path / "sim", "n", 100)
    sess.start()
    out = tmp_path / "r0"
    first = C.play(FakeClient(sess), out, series(), lambda *a: None, reset=False)
    second = C.play(FakeClient(sess), tmp_path / "r1", series(), lambda *a: None, reset=True)
    assert (first["episode"], first["resets"], second["episode"], second["resets"]) == (0, 0, 1, 1)
    assert json.loads((out / "episode.json").read_text()) == first and first["result"]["passed"] is True
    assert first["result"]["phases"][0]["name"] == "approach"


def test_play_waits_until_the_command_topics_have_subscribers(tmp_path):
    seen = []

    class Publish:
        def wait_matched(self):
            seen.append("matched")

        def __call__(self, *a):
            seen.append("sent")
    sess = FakeSession(tmp_path / "s.yaml", "mobile_panda", "open_fridge_door", tmp_path / "sim", "n", 100)
    sess.start()
    C.play(FakeClient(sess), tmp_path / "r", series(), Publish(), reset=False)
    assert seen[0] == "matched"


# ------------------------------------------------------------------ the plugin
class NoDocker:
    def __init__(self, names=()):
        self._names, self.removed = list(names), []

    def names(self):
        return self._names

    def domains(self):
        return set()

    def rm(self, name):
        self.removed.append(name)


class InProcess(S.SimRollout):
    """The sim plugin over FakeSessions; the client container is the client loop in this process."""
    faults, passes, client_rc, client_error = None, staticmethod(lambda n: True), 0, None

    def __init__(self, params=None, **attrs):
        super().__init__(spec, params or {})
        self.__dict__.update(attrs)
        self.sessions, self.client_calls = [], []

    def domain_pool(self):
        return DomainPool(self.prefix, docker=NoDocker())

    def session(self, item, sim_out, name, domain, gui):
        s = FakeSession(item.scene, "mobile_panda", "open_fridge_door", sim_out, name, domain, self.record, gui=gui,
                        passes=self.passes, faults=self.faults)
        self.sessions.append(s)
        return s

    def run_client(self, session, rollout, args, reset, log):
        self.client_calls.append((args, reset))
        if self.client_error:
            raise self.client_error
        if self.client_rc == 0:
            C.play(session.connect(), rollout, series(), lambda *a: None, reset)
        return self.client_rc


@pytest.fixture
def item(tmp_path):
    return EvalItem(Config("train", 3), tmp_path / "scene_3" / "scene.yaml", ({"passed": True, "bag": "/b/0"},))


def run(r, item, tmp_path, name="3_r0", policy=oracle):
    return r.run(item, tmp_path / "set" / name, seed=3, runner=RunnerSpec(), policy=policy, ckpt=None)


def test_one_sim_per_config_reset_between_rollouts_released_at_the_end(item, tmp_path):
    r = InProcess()
    a, b = run(r, item, tmp_path), run(r, item, tmp_path, "3_r1")
    (sim,) = r.sessions
    assert sim.launches == 1 and sim.out == (tmp_path / "set" / "3_sim").resolve() and sim.record == "images"
    assert [c[1] for c in r.client_calls] == [False, True] and r.client_calls[0][0] == ["--bag", "/b/0"]
    assert (a["episode"], a["resets"], b["episode"], b["resets"]) == (0, 0, 1, 1)
    assert a["passed"] and a["phases"] == {"approach": "passed", "grasp": "passed", "open": "passed"} and a["error"] is None
    assert a["failed_phase"] is None and a["reason"] is None and a["bag"].endswith("episode_0_bag")
    assert json.loads((tmp_path / "set" / "3_r0" / "rollout.json").read_text()) == a
    r.release(item)
    assert sim.closed == 1 and not r.scenes and r.pool._leased == {}
    run(r, item, tmp_path, "3_r2")                                              # a new sim after the release
    assert len(r.sessions) == 2 and r.client_calls[-1][1] is False


def test_a_failed_episode_is_the_evaluators_verdict(item, tmp_path):
    rec = run(InProcess(passes=lambda n: False), item, tmp_path)
    assert (rec["passed"], rec["failed_phase"], rec["reason"], rec["error"]) == (False, "approach", "timeout_s", None)
    assert rec["phases"] == {"approach": "failed", "grasp": "not_reached", "open": "not_reached"}


def test_a_sim_fault_relaunches_the_sim_and_runs_the_episode_again(item, tmp_path, capsys):
    r = InProcess(faults={"wait_result": ["result_timeout"], "wait_ready": ["ready_timeout"]})
    rec = run(r, item, tmp_path)
    (sim,) = r.sessions
    assert rec["passed"] and sim.launches == 3 and len(r.client_calls) == 3     # ready fault, result fault, then ok
    assert [c[1] for c in r.client_calls] == [False, False, False] and "relaunch 2/3" in capsys.readouterr().err


def test_the_infrastructure_retries_run_out_into_an_error_record(item, tmp_path):
    rec = run(InProcess({"infra_retries": 1}, faults={"wait_ready": ["ready_timeout"] * 3}), item, tmp_path)
    assert rec["passed"] is False and rec["phases"] is None and rec["error"].startswith("infrastructure: ")


def test_a_client_over_its_wall_limit_or_failing_or_killed_is_an_error_and_the_sim_is_dropped(item, tmp_path):
    r = InProcess({"wall_s": 90}, client_error=subprocess.TimeoutExpired("c", 90))
    rec = run(r, item, tmp_path)
    assert rec["error"].startswith("wall limit 90 s") and r.scenes[str(item.scene)].running is False
    assert run(InProcess(client_rc=1), item, tmp_path)["error"].startswith("the policy client exited with 1")
    killed = InProcess(client_rc=137)
    killed.stopped.set()
    assert run(killed, item, tmp_path)["error"].startswith("interrupted") and killed.client_calls == []
    late = InProcess()
    late.run_client = lambda *a: (late.stopped.set(), 137)[1]                    # the abort kills the client mid-episode
    assert run(late, item, tmp_path)["error"].startswith("interrupted: the client was killed")


def test_a_client_that_leaves_no_episode_file_is_an_error(item, tmp_path):
    r = InProcess()
    r.run_client = lambda *a: 0
    assert "left no episode.json" in run(r, item, tmp_path)["error"]


def test_stop_removes_the_runs_containers_and_starts_no_sim(item, tmp_path):
    r = InProcess()
    r.pool.docker = NoDocker([f"{r.prefix}-0", f"{r.prefix}-0-client", "ev-other-0", "revolve-sim"])
    r.stop()
    assert r.pool.docker.removed == [f"{r.prefix}-0", f"{r.prefix}-0-client"]      # only this run's
    assert run(r, item, tmp_path)["error"].startswith("interrupted") and r.client_calls == []


def test_oracle_without_a_bag_raises(item, tmp_path):
    with pytest.raises(EvalError, match="no passed datagen episode"):
        run(InProcess(), EvalItem(Config("heldout", 3), tmp_path / "s.yaml"), tmp_path)


def test_params_validation_and_policy_check():
    with pytest.raises(EvalError, match="unknown parameters.*speed"):
        REGISTRY.create("rollout", "sim", spec, {"speed": 2})
    with pytest.raises(EvalError, match="record must be"):
        REGISTRY.create("rollout", "sim", spec, {"record": "video"})
    r = REGISTRY.create("rollout", "sim", spec, {"record": "no-images", "wall_s": 30, "ready_s": 60})
    assert (r.record, r.wall_s, r.ready_s, r.infra_retries, r.default_workers(oracle)) == ("no-images", 30.0, 60.0, 3, 1)
    r.validate(oracle)
    with pytest.raises(SpecError, match="policy 'smolvla' is not available on the new simulator.*Rosetta contract"):
        r.validate(smolvla)
    assert r.prefix == sim_prefix(spec) and r.prefix.startswith("ev-")


def test_the_client_container_command(item, tmp_path):
    seen = {}

    class Container:
        def run(self, log, *, timeout, watch):
            seen.update(log=log, timeout=timeout, fault=watch())
            return 0

    class Session:
        out = tmp_path / "sim"

        def client(self, command, **kw):
            seen.update(command=command, **kw)
            return Container()

        def check(self):
            return "fault"
    r = REGISTRY.create("rollout", "sim", spec, {"wall_s": 90, "ready_s": 60})
    bag = REPO / "runs" / "x" / "episode_0_bag"
    assert r.run_client(Session(), tmp_path / "r0", ["--bag", str(bag)], True, tmp_path / "c.log") == 0
    assert seen["command"] == ["python3", "-m", "evaluation.rollout_client", "--sim-out", str(tmp_path / "sim"), "--rollout",
                               str(tmp_path / "r0"), "--bag", "/repo/runs/x/episode_0_bag", "--ready-s", "60.0", "--lead-s", "0.1",
                               "--reset"]
    assert seen["pythonpath"] == [S.CLIENT_SRC] and seen["mounts"] == [tmp_path / "r0"] and seen["timeout"] == 90.0
    assert seen["fault"] == "fault"
    r.stopped.set()
    r.run_client(Session(), tmp_path / "r0", [], False, tmp_path / "c.log")
    assert seen["fault"] is None and "--reset" not in seen["command"]            # a stopped run does not relaunch sims
    assert S.in_container("/elsewhere/x") == "/elsewhere/x" and S.in_container("--bag") == "--bag"
