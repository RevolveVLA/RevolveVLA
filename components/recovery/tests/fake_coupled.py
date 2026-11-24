"""Fakes for the coupled evaluation + recovery tests: an evaluation of scripted rollouts that leave what the real `sim` rollout
leaves (results with fake bags that `fake_recovery.read_messages` reads), and a generator that logs when its jobs run.

`TRIES[seed]` = how that config's episode goes (fake_recovery.Try; default passes). `EVENTS` = what ran, in order; `PEAK` = the most
rollouts + jobs that ran at the same time."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import fake_recovery as F
import yaml
from datagen.core.interfaces import Config
from evaluation.core.interfaces import Built, EvalItem, EvalSet, Policy, Rollout
from evaluation.core.registry import REGISTRY as EVAL
from revolve_common import specs

TRIES: dict[int, F.Try] = {}
EVENTS: list[str] = []
PEAK = [0]
_LIVE, _LOCK = [0], threading.Lock()


def reset():
    TRIES.clear(), EVENTS.clear(), F.CALLS.clear()
    PEAK[0] = _LIVE[0] = 0


class busy:
    """`with busy("what"):` logs start / end and tracks how many run at once."""

    def __init__(self, what: str, delay: float = 0.02):
        self.what, self.delay = what, delay

    def __enter__(self):
        with _LOCK:
            _LIVE[0] += 1
            PEAK[0] = max(PEAK[0], _LIVE[0])
            EVENTS.append(f"start {self.what}")
        time.sleep(self.delay)

    def __exit__(self, *exc):
        with _LOCK:
            _LIVE[0] -= 1
            EVENTS.append(f"end {self.what}")


@EVAL.register("set", "fake_scenes")
class Scenes(EvalSet):
    """The first `count` seeds of the pool, each a kitchen scene instance starting beside the fridge."""

    def __init__(self, spec):
        self.spec = spec

    def build(self, set_spec, out, workers, abort=None):
        base = yaml.safe_load((specs.REPO_SPECS / "scenes" / "kitchen_partnet" / "scene.yaml").read_text())
        items = []
        for seed in set_spec.pool.seeds()[:set_spec.count]:
            scene = Path(out) / set_spec.name / f"scene_{seed}" / "scene.yaml"
            scene.parent.mkdir(parents=True, exist_ok=True)
            scene.write_text(yaml.safe_dump(base | {"robot_start": {"base": [-1.75 + 0.01 * (seed % 10), 1.1, 1.5], "arm": "folded"}}))
            items.append(EvalItem(Config(set_spec.name, seed), scene))
        return Built(items)


@EVAL.register("policy", "fake_policy")
class FakePolicy(Policy):
    name, needs_ckpt, client_only = "fake_policy", False, True

    def __init__(self, spec, params):
        pass


@EVAL.register("rollout", "fake_bags")
class BagRollout(Rollout):
    """One scripted episode: the sim folder `<set>/<seed>_sim/` gets `episode_<n>.json` and a bag of `messages.json`."""

    def __init__(self, spec, params):
        self.delay = params.get("delay_s", 0.02)

    def run(self, item, out, *, seed, runner, policy, ckpt, gui=False):
        with busy(f"rollout {seed}", self.delay):
            out.mkdir(parents=True, exist_ok=True)
            sim, t = out.parent / f"{seed}_sim", TRIES.get(seed, F.Try(seed=seed))
            sim.mkdir(exist_ok=True)
            n = len(list(sim.glob("episode_*.json")))
            res = F.result(t, n, sim, "kitchen_partnet")
            start = yaml.safe_load(Path(item.scene).read_text())["robot_start"]["base"]
            (sim / f"episode_{n}_bag").mkdir()
            (sim / f"episode_{n}_bag" / "messages.json").write_text(json.dumps(F.messages(t, n, start)))
            (sim / f"episode_{n}.json").write_text(json.dumps(res))
            record = dict(seed=seed, dir=str(out), policy=policy.name, ckpt=None, scene=str(item.scene), episode=n, resets=n, steps=None,
                          phases={p["name"]: p["status"] for p in res["phases"]}, failed_phase=res["failed_phase"], reason=res["reason"],
                          check=None, passed=res["passed"], duration_s=res["duration_s"], bag=res["bag"], result_file=res["result_file"],
                          error=None)
            (out / "rollout.json").write_text(json.dumps(record))
            return record


class ProbeGenerator(F.FakeGenerator):
    """The fake generator that counts as busy while a job drives."""

    def drive(self, client, instance, task, robot):
        with busy(f"job {Path(instance).parent.name}"):
            return super().drive(client, instance, task, robot)


from datagen.core.registry import register  # noqa: E402

register("generator", "probe_recovery")(ProbeGenerator)
