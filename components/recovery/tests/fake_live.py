"""Fakes for the live recorder tests: a sim client in this process whose clock and /evaluator/state follow a script (what a policy's
loop and a supervisor see), a state source for the robot's joints and a fake expert (registry kind `expert`).

Timeline of an episode (sim s since its start): approach from 0, grasp from 6, open from 9; the result is out at `Script.end_s`
(a failed episode, as the evaluator's timeout) or at once when the expert has finished (a passed one). The clock moves 10 ms
per `now()` call: the policy's stream loop polls it."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from unittest import mock

import numpy as np
from datagen.core.interfaces import Config
from evaluation import rollout_client as C
from evaluation.core.interfaces import Built, EvalItem, EvalSet, Rollout
from evaluation.core.registry import REGISTRY as EVAL
from evaluation.oracle_actions import ActionSeries
from recovery.core.registry import REGISTRY
from revolve_common.sim_fake import FakeClient, FakeSession

START = 3.9                             # the fake results' sim_time_start
PHASES = (("approach", 0.0), ("grasp", 6.0), ("open", 9.0))
LOG: list = []                          # what the fakes did, in order
STREAMED: list = []                     # what a streaming policy's publisher was given: (joint targets, base twist)


class Streamer:
    """The publisher a policy client makes for its actions (instead of the ROS one): it logs what it is given."""

    def __init__(self, node, series, lead_s):
        pass

    def wait_matched(self):
        pass

    def __call__(self, pos, base):
        STREAMED.append((pos, base))


@dataclass
class Script:
    end_s: float = 20.0
    values: Mapping = field(default_factory=lambda: {"open.while[1]": 0.0})     # /evaluator/state values (by name) at any time
    moving: bool = False                # do the robot's joints move (else it stands still: `stall`)
    only_phase: str | None = None       # the evaluator stays in this phase
    expert_works: bool = True           # the expert's takeover finishes the episode
    passes_alone: bool = False          # the policy itself passes the episode at `end_s`

    def phase(self, t: float) -> str:
        return self.only_phase or [p for p, s in PHASES if s <= t][-1]


class LiveClient(FakeClient):
    def __init__(self, session, script: Script):
        super().__init__(session)
        self.script, self.t, self.t0, self.finished, self.episode, self._k, self._state = script, START, None, False, None, -1, None
        self.node = self                                        # StateStub reads the client's joints through its "node"

    def start_episode(self) -> int:
        self.episode = super().start_episode()
        self.t0, self.finished, self._k = self.t, False, -1
        LOG.append(("start", self.episode))
        return self.episode

    def now(self) -> float:
        self.t += 0.01
        return self.t

    def state(self):
        if self.t0 is None:
            return None
        k = int((self.t - self.t0) * 10 + 1e-9)
        if k != self._k:                                        # a new message every 0.1 s: a new object
            self._k = k
            at = k / 10
            self._state = {"t": round(self.t0 + at, 3), "episode": self.episode, "phase": self.script.phase(at),
                           "values": dict(self.script.values), "resets": 0}
        return self._state

    def has_result(self, n: int) -> bool:
        if self.t0 is not None and self.t - self.t0 >= self.script.end_s:
            self.finished = self.finished or self.script.passes_alone
            return True
        return self.finished

    def joints(self):
        k = (self.t - self.t0) if self.script.moving and self.t0 is not None else 0.0
        return {"panda_joint1": 0.5 * k, "panda_finger_joint1": 0.04}

    def base(self):
        return (0.0, 0.0, 0.0)


class StateStub:
    """Stands in for plugins/live/ros_state.RosState: the joints / base of the client its `node` is."""

    def __init__(self, node, robot):
        self.joints, self.base = node.joints, node.base


class FakeExpert:
    """expert fake_takeover: finishes the episode at once (the result is out; passed) unless its `fail` param says it cannot.
    With `drives` it drives until `ctx.stop()` says to hand back (then it returns, the episode runs on) or the result is out."""

    def __init__(self, spec, params):
        self.fail, self.raises, self.drives = params.get("fail", False), params.get("raises", False), params.get("drives", False)

    def prepare(self, ctx):
        LOG.append(("prepare", ctx.task))

    def takeover(self, ctx, why):
        LOG.append(("takeover", why))
        if self.raises:
            raise RuntimeError("planner died")
        if self.drives:
            while not ctx.stop() and not ctx.client.has_result(0):
                ctx.client.now()
            if ctx.stop():
                LOG.append(("stopped", round(ctx.client.t - ctx.client.t0, 1)))
                return {"expert": "fake_v1", "notes": "stopped", "max_delta_action": 0.02}
            return {"expert": "fake_v1", "notes": "ok", "max_delta_action": 0.02}
        ctx.client.finished = not self.fail and ctx.client.script.expert_works
        return {"expert": "fake_v1", "notes": "ok", "max_delta_action": 0.02}


REGISTRY.register("expert", "fake_takeover")(FakeExpert)


# ------------------------------------------------------------------ an evaluation of fakes
SCRIPTS: dict[int, Script] = {}         # per config seed: how its episode goes (default: Script())
SERIES = ActionSeries({"arm": ["j"]}, {"arm": np.arange(1200, dtype=float)[:, None]}, np.zeros((1200, 3)))


@EVAL.register("set", "fake_set")
class FakeSet(EvalSet):
    """The first `count` seeds of the set's pool as configs; each has a recorded (passed) bag for the oracle-based policies."""

    def __init__(self, spec):
        self.spec = spec

    def build(self, set_spec, out, workers, abort=None):
        items = []
        for seed in set_spec.pool.seeds()[:set_spec.count]:
            scene = Path(out) / set_spec.name / f"scene_{seed}" / "scene.yaml"
            scene.parent.mkdir(parents=True, exist_ok=True)
            scene.write_text("name: fake\n")
            items.append(EvalItem(Config(set_spec.name, seed), scene, ({"passed": True, "bag": "/fake/bag"},)))
        return Built(items)


@EVAL.register("rollout", "fake_client")
class ClientRollout(Rollout):
    """What the `sim` rollout does around a sim, in this process: the policy's REAL client (evaluation.rollout_client.main with the
    arguments of the policy) against a scripted client; its bag read and its publisher are stubs."""

    def __init__(self, spec, params):
        pass

    def run(self, item, out, *, seed, runner, policy, ckpt, gui=False):
        out.mkdir(parents=True, exist_ok=True)
        session = FakeSession(item.scene, "mobile_panda", "open_fridge_door", out.parent / f"{seed}_sim", "n", 100)
        session.start()
        client = LiveClient(session, SCRIPTS.get(seed, Script()))
        session.passes = lambda n: client.finished
        with mock.patch.multiple(C, run_client=lambda fn, sim_out: fn(client), read_bag=lambda bag: SERIES,
                                 RosStreamer=Streamer):
            C.main(["--sim-out", str(session.out), "--rollout", str(out), *policy.client_args(item), "--ready-s", "1"])
        ep = json.loads((out / "episode.json").read_text())
        res = ep["result"]
        record = dict(seed=seed, dir=str(out), policy=policy.name, ckpt=None, scene=str(item.scene), episode=ep["episode"],
                      resets=ep["resets"], steps=ep.get("steps"), phases={p["name"]: p["status"] for p in res["phases"]},
                      failed_phase=res["failed_phase"], reason=res["reason"], check=res["check"], passed=bool(res["passed"]),
                      duration_s=res["duration_s"], bag=res["bag"], result_file=res["result_file"], error=None)
        (out / "rollout.json").write_text(json.dumps(record))
        return record
