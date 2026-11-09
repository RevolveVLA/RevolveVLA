"""Stub components for the manager integration tests: steps.py runs this instead of the real step when
REVOLVE_STUB=<script.json> is set. The step container, wrapper, HTTP polling, labels, kill and resume are real;
only the work is faked: progress lines, then the files the manager reads (manifest.json, start.json,
meta/revolve_dataset.json, train.json + config.json, eval_summary.json) in the components' formats.

script.json:
  duration_s          seconds of PROGRESS lines per step (default 1.5); "durations": {"t1/train": 8}
  fail                {"t1/train": 1}: the first N attempts exit 1
  hang                {"t0/dataset": 1}: the first N attempts start a child container, print once, then hang
  verdicts            {"<turn>": {"heldout"|"probe": {"<seed>": "<failed phase>"}}}: unlisted seeds pass
Every call is appended to <exp dir>/stub_calls.jsonl (step, turn, attempt).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from revolve_common import job as J

REPO = Path(__file__).resolve().parents[2]
PHASES = ("approach", "grasp", "open", "release")


def arg(argv, flag):
    return argv[argv.index(flag) + 1]


def ticks(seconds, train=False):
    n = max(1, int(seconds / 0.25))
    for i in range(n):
        print("PROGRESS " + json.dumps(dict(step=(i + 1) * 10, total_steps=n * 10, loss=round(1.0 / (i + 1), 4))
                                       if train else dict(tick=i + 1, of=n)), flush=True)
        time.sleep(0.25)


def run(p, job_dir):
    script = json.loads(Path(os.environ["REVOLVE_STUB"]).read_text())
    key = f"t{p['turn']}/{p['step']}"
    exp_dir = REPO / "runs" / p["exp"]
    with open(exp_dir / "stub_calls.jsonl", "a") as f:
        f.write(json.dumps(dict(step=p["step"], turn=p["turn"], attempt=p["attempt"], t=time.time())) + "\n")
    if p["attempt"] < script.get("hang", {}).get(key, 0):
        import docker
        docker.from_env().containers.run("revolve-manager:latest", ["sleep", "600"], detach=True,
                                         name=f"{J.read_job(job_dir).id}-child")
        print("stub: hanging", flush=True)
        time.sleep(3600)
    ticks(script.get("durations", {}).get(key, script.get("duration_s", 1.5)), train=p["step"] == "train")
    if p["attempt"] < script.get("fail", {}).get(key, 0):
        print("stub: scripted failure", flush=True)
        return 1
    globals()[p["step"].split("_")[0]](p, script)
    return 0


def episode(d, **ep):
    d.mkdir(parents=True, exist_ok=True)
    J.write_json(d / "episode.json", ep)


def generate(p, script):
    a = p["argv"]
    out = Path(arg(a, "--out"))
    names = [f"seed_{s}_a0" for s in arg(a, "--seeds").split(",")]
    for n in names:
        episode(out / "episodes" / n, task="stub", datagen=dict(seed=int(n.split("_")[1]), label="demo"))
    J.write_json(out / "manifest.json", dict(episodes=names, counts=dict(episodes_accepted=len(names))))


def starts(p, script):
    a = p["argv"]
    for s in arg(a, "--seeds").split(","):
        J.write_json(Path(arg(a, "--out")) / f"seed_{s}" / "start.json", dict(seed=int(s), dir="stub"))


def recover(p, script):
    a = p["argv"]
    out = Path(arg(a, "--out"))
    rollouts = a[a.index("--rollouts") + 1:a.index("--out")]
    n = int(arg(a, "--episodes-per-failure"))
    eps = []
    for r in rollouts:
        seed = json.loads((Path(r) / "episode.json").read_text())["seed"]
        for i in range(n):
            d = Path(Path(r).name) / "episodes" / f"ep_{i}"
            episode(out / d, task="stub", recovery=dict(source="recovery", rollout_seed=seed, from_rollout=r, suffix=True))
            eps.append(str(d))
    J.write_json(out / "manifest.json", dict(episodes=eps, counts=dict(episodes_accepted=len(eps))))


def dataset(p, script):
    d = Path(p["turn_dir"]) / "dataset"
    prev = json.loads((Path(p["prev_dataset"]) / "stub_episodes.json").read_text()) if p["prev_dataset"] else []
    eps = prev + list(p["episodes"])
    J.write_json(d / "stub_episodes.json", eps)
    J.write_json(d / "meta" / "revolve_dataset.json", dict(episodes=len(eps), frames=100 * len(eps), inputs=eps))


def train(p, script):
    out = Path(p["out"])
    J.write_json(out / "config.json", dict(type="stub"))
    J.write_json(out / "train.json", dict(steps=p["training"]["steps"], mode=p["training"]["mode"], dataset=p["dataset"],
                                         prev=p["prev"], losses=[[10, 1.0], [20, 0.5]]))


def eval(p, script):          # noqa: A001  (the step is eval_heldout / eval_probe)
    a = p["argv"]
    name = arg(a, "--set")
    out = Path(arg(a, "--out")) / name
    fails = script.get("verdicts", {}).get(str(p["turn"]), {}).get(name, {})
    table = {}
    for s in arg(a, "--seeds").split(","):
        d = out / f"{s}_r0"
        episode(d, task="stub", seed=int(s), policy_rollout={})
        f = fails.get(s)
        ph = {x: "pass" for x in PHASES}
        if f:
            i = PHASES.index(f)
            ph = {x: ("pass" if j < i else "fail" if j == i else "not_reached") for j, x in enumerate(PHASES)}
        table[s] = {"r0": dict(dir=str(d.relative_to(REPO)), first_try=ph, passed_first_try=not f, failed_phase=f,
                               error=None, passed=not f)}
    k = sum(not fails.get(s) for s in table)
    J.write_json(out / "eval_summary.json", dict(set=name, rollouts=len(table), first_try_passed=k, all_pass=k == len(table),
                                                 all_pass_with_retries=k == len(table), table=table))


if __name__ == "__main__":
    sys.exit("run through manager.steps")
