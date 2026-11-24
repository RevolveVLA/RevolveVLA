"""Fakes for the recovery CPU tests: a finished evaluation run (`write_eval_run`: results.json, the scene instances, the sim's
episode results and fake bags), the bag reader that reads those bags (`read_messages`) and a datagen generator (`fake_recovery`).

An eval try is scripted by `Try`: which phase failed and when, whether the robot stalls in approach, whether the door has moved.
Timeline of an episode (sim s since its start; the episode's sim_time_start is T0): approach from 0, grasp from 6, open from 9
(each only when reached); the base drives and the arm moves during approach, the door opens by 0.1 rad/s during open."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from datagen.core.interfaces import Generator, Result
from datagen.core.registry import register
from revolve_common import specs
from revolve_common.supervision import BagMessage

T0 = 3.9
ARM = [f"panda_joint{i}" for i in range(1, 8)]
HAND = "panda_finger_joint1"
PHASE_START = {"approach": 0.0, "grasp": 6.0, "open": 9.0}
T_FAIL = {"approach": 12.0, "grasp": 11.0, "open": 24.0}
ROBOT = "mobile_panda"
CALLS = []              # ("drive", scene file) of the fake generator, in order


@dataclass
class Try:
    set: str = "heldout"
    seed: int = 100
    rollout: int = 0
    failed: str | None = None           # the failed phase; None: the episode passed
    t_fail: float | None = None
    stall: bool = False                 # the robot stands still during approach
    door: float | None = None           # the door joint when the failure happens (default: 0 before open, then 0.1 / s)

    @property
    def fail_t(self) -> float:
        return self.t_fail or T_FAIL[self.failed]

    def phase_at(self, t: float) -> str:
        reached = [p for p, s in PHASE_START.items() if s <= t and (self.failed is None or PHASE_START[p] <= PHASE_START[self.failed])]
        return reached[-1]

    def door_at(self, t: float) -> float:
        if self.door is not None:
            return self.door
        return max(0.0, 0.1 * (t - PHASE_START["open"])) if self.failed in (None, "open") else 0.0

    def base_at(self, t: float, start) -> list[float]:
        f = 0.0 if self.stall else min(t, 6.0) / 6.0
        return [start[0], start[1] + 0.8 * f, start[2]]

    def arm_at(self, t: float) -> list[float]:
        f = 0.0 if self.stall else min(t, 6.0) / 6.0
        return [0.1 * (i + 1) + 0.5 * f for i in range(7)]


def messages(t: Try, episode: int, start) -> list:
    """The bag's messages: state, joints, pose and the door at 10 Hz until the end of the episode."""
    end, out = t.fail_t if t.failed else 14.0, []
    for k in range(int(end * 10) + 1):
        s = k / 10
        a = T0 + s
        values = {"approach.done[0]": 0.3, "approach.done[1]": 0.0, "approach.done[2]": 0.0, "grasp.done[0]": 0.0,
                  "open.done[0]": t.door_at(s), "open.while[0]": 2.0, "open.while[1]": 0.0}
        out += [["/evaluator/state", a, json.dumps({"t": a, "ready": True, "resets": 0, "episode": episode, "phase": t.phase_at(s),
                                                      "values": values})],
                ["/joint_states", a, {"name": ARM + [HAND], "position": t.arm_at(s) + [0.04]}],
                [f"/model/{ROBOT}/pose", a, t.base_at(s, start)],
                ["/world/kitchen_partnet/model/fridge/joint_state", a, {"name": ["top_door_hinge"], "position": [t.door_at(s)]}]]
    return out


def read_messages(bag, wanted):
    """The bag reader of `BagSource` over the fake bags (messages.json)."""
    return [BagMessage(topic, at, data) for topic, at, data in json.loads((Path(bag) / "messages.json").read_text()) if wanted(topic)]


def result(t: Try, episode: int, sim: Path, seed_scene: str) -> dict:
    reached = [p for p in PHASE_START if t.failed is None or PHASE_START[p] <= PHASE_START[t.failed]]
    phases = [{"name": p, "status": "passed" if t.failed is None else "failed" if p == t.failed else
               "passed" if p in reached and p != t.failed else "not_reached",
               "t_start": PHASE_START[p] if p in reached else None, "t_end": None, "reason": None, "last": {}} for p in PHASE_START]
    return {"task": "open_fridge_door", "scene": seed_scene, "robot": ROBOT, "episode": episode, "sim_time_start": T0,
            "passed": t.failed is None, "failed_phase": t.failed, "reason": None if t.failed is None else "timeout_s",
            "t_fail": t.fail_t if t.failed else None, "check": None, "condition": None, "value": None,
            "duration_s": t.fail_t if t.failed else 14.0, "phases": phases, "sanity_last": {}, "contacts": [],
            "result_file": str(sim / f"episode_{episode}.json"), "bag": str(sim / f"episode_{episode}_bag")}


def write_eval_run(root: Path, tries: list[Try]) -> Path:
    """`<root>/eval/`: an evaluation run with the given tries (several tries of one rollout: same set, seed and rollout)."""
    run = root / "eval"
    base = yaml.safe_load((specs.REPO_SPECS / "scenes" / "kitchen_partnet" / "scene.yaml").read_text())
    sets, numbers = {}, {}
    for t in tries:
        scene = run / t.set / f"scene_{t.seed}" / "scene.yaml"
        start = [-1.75 + 0.01 * (t.seed % 10), 1.1, 1.5]
        scene.parent.mkdir(parents=True, exist_ok=True)
        scene.write_text(yaml.safe_dump(base | {"robot_start": {"base": start, "arm": "folded"}}, sort_keys=False))
        sim = run / t.set / f"{t.seed}_sim"
        n = numbers[t.set, t.seed] = numbers.get((t.set, t.seed), -1) + 1
        res = result(t, n, sim, "kitchen_partnet")
        bag = sim / f"episode_{n}_bag"
        bag.mkdir(parents=True, exist_ok=True)
        (bag / "messages.json").write_text(json.dumps(messages(t, n, start)))
        (sim / f"episode_{n}.json").write_text(json.dumps(res))
        key = f"{t.seed}_r{t.rollout}"
        record = {"seed": t.seed, "dir": str(run / t.set / key), "policy": "oracle", "ckpt": None, "scene": str(scene), "episode": n,
                  "phases": {p["name"]: p["status"] for p in res["phases"]}, "failed_phase": t.failed, "reason": res["reason"],
                  "check": None, "passed": t.failed is None, "duration_s": res["duration_s"], "bag": res["bag"],
                  "result_file": res["result_file"], "error": None}
        entry = sets.setdefault(t.set, {"rollouts": {}})["rollouts"].setdefault(key, {"seed": t.seed, "passed": False, "tries": []})
        entry["tries"].append({"dir": record["dir"], "passed": t.failed is None, "detail": {}, "termination": "timeout_s",
                               "record": record})
        entry["passed"] = t.failed is None
    run.mkdir(parents=True, exist_ok=True)
    (run / "results.json").write_text(json.dumps({"spec_hash": "fake", "ckpt": None, "sets": sets}))
    return run


@register("generator", "fake_recovery")
class FakeGenerator(Generator):
    def __init__(self, spec, params):
        self.spec, self.bad_plans = spec, set(params.get("bad_plans", ()))

    def plan(self, instance, task, robot):
        CALLS.append(("plan", Path(instance)))
        return Result.fail("no plan") if Path(instance).parent.name in self.bad_plans else Result()

    def drive(self, client, instance, task, robot):
        CALLS.append(("drive", Path(instance)))
        return {"expert": "fake_v1", "notes": ""}
