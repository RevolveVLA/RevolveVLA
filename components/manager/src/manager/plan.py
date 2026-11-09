"""The experiment loop as data: which step comes next, each step's inputs, what it produced, summary.json.
No docker here (unit-tested on its own); loop.py runs the steps.

  turn 0:  generate (train seeds) -> starts (probe + held-out, once) -> dataset -> train -> eval_heldout -> eval_probe
  turn k:  recover (turn k-1's probe failures) -> dataset (merged after turn k-1's) -> train -> eval_heldout -> eval_probe
  stop:    turn k's held-out eval all_pass (every seed passes every phase on the first try), or k + 1 == max_turns.
           eval_probe is skipped when no turn follows (its failures would feed nothing).

state.json (runs/<exp>/, atomic): {exp, spec, spec_sha256, status, stop_reason, error, turns: [{turn, steps: {name:
{state, attempts: [{n, container, job_dir, state, ...}], outputs}}}]}. Step states: pending running done skipped failed.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from revolve_common import job as J

TURN0 = ("generate", "starts", "dataset", "train", "eval_heldout", "eval_probe")
TURNK = ("recover", "dataset", "train", "eval_heldout", "eval_probe")
FINISHED = ("done", "skipped")


class HeldoutLeak(AssertionError):
    pass


def steps_of(k):
    return TURN0 if k == 0 else TURNK


def read(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else None


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def new_state(exp_name, spec_path):
    return dict(exp=exp_name, spec=str(spec_path), spec_sha256=sha256(spec_path), status="running", stop_reason=None,
                error=None, created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), turns=[new_turn(0)])


def new_turn(k):
    return dict(turn=k, steps={n: dict(state="pending", attempts=[], outputs={}) for n in steps_of(k)})


# ------------------------------------------------------------------ the next action
def next_action(state, max_turns):
    """-> ("run", k, step) | ("skip", k, step, reason) | ("turn", k) | ("stop", reason)."""
    t = state["turns"][-1]
    k = t["turn"]
    for name in steps_of(k):
        s = t["steps"][name]
        if s["state"] in FINISHED:
            continue
        if s["state"] == "failed":
            return ("stop", "failed")
        if name == "eval_probe":
            if t["steps"]["eval_heldout"]["outputs"].get("all_pass"):
                return ("skip", k, name, "held-out all pass: the experiment stops")
            if k + 1 >= max_turns:
                return ("skip", k, name, "last turn: no recovery follows")
        return ("run", k, name)
    if t["steps"]["eval_heldout"]["outputs"].get("all_pass"):
        return ("stop", "all_pass")
    if k + 1 >= max_turns:
        return ("stop", "max_turns")
    return ("turn", k + 1)


# ------------------------------------------------------------------ paths and inputs
class Paths:
    def __init__(self, exp_dir):
        self.exp = Path(exp_dir)
        self.starts = self.exp / "starts"

    def turn(self, k):
        return self.exp / f"turn_{k}"

    def data(self):
        return self.turn(0) / "data"

    def recovery(self, k):
        return self.turn(k) / "recovery"

    def ckpt(self, k):
        return self.turn(k) / "ckpt"

    def eval(self, k):
        return self.turn(k) / "eval"

    def step_dir(self, k, step, n):
        return self.turn(k) / "steps" / f"{step}_a{n}"


def seed_set(r):
    return {s: getattr(r.seeds, s).seeds() for s in ("train", "probe", "heldout")}


def episode_seed(d):
    """The seed an episode came from: a demo's datagen seed, a recovery episode's rollout seed."""
    ep = read(Path(d) / "episode.json") or {}
    if "recovery" in ep:
        return ep["recovery"].get("rollout_seed")
    return (ep.get("datagen") or {}).get("seed", ep.get("seed"))


def check_training_episodes(episodes, exp, paths):
    """Held-out data never reaches training: no episode from a held-out seed, none from an eval/heldout dir."""
    held = set(seed_set(exp)["heldout"])
    for d in episodes:
        d = Path(d)
        if any(p.name == "heldout" and p.parent.name == "eval" for p in d.parents):
            raise HeldoutLeak(f"{d}: under a held-out eval dir")
        s = episode_seed(d)
        if s is None or int(s) in held:
            raise HeldoutLeak(f"{d}: seed {s} is held-out or unknown (held-out {sorted(held)})")
        if not d.resolve().is_relative_to(paths.exp.resolve()):
            raise HeldoutLeak(f"{d}: not under {paths.exp}")


def probe_failures(state, k, exp, paths):
    """Turn k's probe rollouts that failed a phase on the first try (dirs = recovery's input)."""
    probe = set(seed_set(exp)["probe"])
    out = []
    for f in state["turns"][k]["steps"]["eval_probe"]["outputs"].get("failures", []):
        d = paths.exp.parents[1] / f["dir"]               # eval_summary dirs are repo-relative
        if not d.resolve().is_relative_to((paths.eval(k) / "probe").resolve()) or int(f["seed"]) not in probe:
            raise HeldoutLeak(f"recovery input {d} (seed {f['seed']}) is not a turn-{k} probe rollout")
        out.append(str(d))
    return out


def step_params(state, k, step, r, paths):
    """The step's params (job.json) or None when there is nothing to do (the step is skipped)."""
    e, task = r.experiment, r.task.name
    sd = seed_set(e)
    if step == "generate":
        return dict(script="components/datagen/generate.py", watch=str(paths.data()),
                    argv=["--task", task, "--seeds", ",".join(map(str, sd["train"])), "--out", str(paths.data()),
                          "--episodes-per-seed", str(e.datagen.episodes_per_seed), "--workers", str(e.datagen.workers)])
    if step == "starts":
        return dict(script="components/evaluation/starts.py", watch=str(paths.starts),
                    argv=["--task", task, "--seeds", ",".join(map(str, sd["heldout"] + sd["probe"])),
                          "--out", str(paths.starts), "--workers", str(e.datagen.workers)])
    if step == "recover":
        rollouts = probe_failures(state, k - 1, e, paths)
        if not rollouts:
            return None
        return dict(script="components/recovery/recover.py", watch=str(paths.recovery(k)),
                    argv=["--task", task, "--rollouts", *rollouts, "--out", str(paths.recovery(k)),
                          "--episodes-per-failure", str(e.recovery.episodes_per_failure),
                          "--phases", ",".join(e.recovery.phases), "--workers", str(e.datagen.workers)])
    if step == "dataset":
        eps = new_episodes(state, k, paths)
        if k > 0 and not eps:
            return None                                    # nothing new: train on turn k-1's dataset
        check_training_episodes(eps, e, paths)
        return dict(task=task, turn_dir=str(paths.turn(k)), turn=k, episodes=eps,
                    prev_dataset=str(dataset_of(state, k - 1)) if k > 0 else None, watch=str(paths.turn(k) / "jobs"))
    if step == "train":
        prev = str(paths.ckpt(k - 1)) if k > 0 else None
        return dict(training=e.training.model_dump(mode="json"), dataset=str(dataset_of(state, k)),
                    out=str(paths.ckpt(k)), prev=prev, watch=str(paths.turn(k) / "jobs"))
    if step in ("eval_heldout", "eval_probe"):
        name = step.split("_", 1)[1]
        return dict(script="components/evaluation/evaluate.py", watch=str(paths.eval(k) / name),
                    argv=["--task", task, "--ckpt", str(paths.ckpt(k)), "--seeds", ",".join(map(str, sd[name])),
                          "--set", name, "--out", str(paths.eval(k)), "--starts", str(paths.starts),
                          "--rollouts-per-seed", str(e.eval.rollouts_per_seed), "--retries", str(e.eval.retries)])
    raise ValueError(step)


def new_episodes(state, k, paths):
    out = state["turns"][k]["steps"]["generate" if k == 0 else "recover"]["outputs"]
    return list(out.get("episodes", []))


def dataset_of(state, k):
    """The dataset turn k trains on: its own, or the last earlier turn's when it had nothing new."""
    for j in range(k, -1, -1):
        d = state["turns"][j]["steps"]["dataset"]["outputs"].get("dataset")
        if d:
            return d
    raise ValueError(f"turn {k}: no dataset")


# ------------------------------------------------------------------ what a finished step produced
def arg(params, flag):
    return params["argv"][params["argv"].index(flag) + 1]


def collect(step, params, paths):
    """-> (outputs, error). error: the step exited 0 but its outputs are not usable."""
    if step == "generate":
        m = read(paths.data() / "manifest.json")
        if not m or not m.get("episodes"):
            return {}, "no accepted episode"
        return dict(episodes=[str(paths.data() / "episodes" / n) for n in m["episodes"]], counts=m.get("counts")), None
    if step == "starts":
        seeds = arg(params, "--seeds").split(",")
        missing = [s for s in seeds if not (paths.starts / f"seed_{s}" / "start.json").is_file()]
        return (dict(seeds=len(seeds)), None) if not missing else ({}, f"no start for seeds {missing}")
    if step == "recover":
        out = Path(arg(params, "--out"))
        m = read(out / "manifest.json")
        if m is None:
            return {}, "no recovery manifest"
        return dict(episodes=[str(out / d) for d in m["episodes"]], counts=m.get("counts")), None
    if step == "dataset":
        d = Path(params["turn_dir"]) / "dataset"
        s = read(d / "meta" / "revolve_dataset.json")
        return (dict(dataset=str(d), episodes=s.get("episodes"), frames=s.get("frames"),
                     new_episodes=len(params["episodes"])), None) if s else ({}, "no dataset summary")
    if step == "train":
        t = read(Path(params["out"]) / "train.json")
        if not t or not (Path(params["out"]) / "config.json").is_file():
            return {}, "no checkpoint"
        losses = t.get("losses") or []
        return dict(ckpt=params["out"], steps=t.get("steps"), mode=t.get("mode"),
                    final_loss=losses[-1][1] if losses and isinstance(losses[-1], list) else t.get("final_loss")), None
    if step.startswith("eval_"):
        s = read(Path(params["watch"]) / "eval_summary.json")
        if s is None:
            return {}, "no eval_summary.json"
        table = {seed: {r: row.get("first_try") for r, row in rows.items()} for seed, rows in s["table"].items()}
        failures = [dict(seed=int(seed), dir=row["dir"], failed_phase=row["failed_phase"])
                    for seed, rows in s["table"].items() for row in rows.values()
                    if row.get("first_try") is not None and not row["passed_first_try"]]
        return dict(all_pass=bool(s["all_pass"]), all_pass_with_retries=s.get("all_pass_with_retries"),
                    first_try_passed=s["first_try_passed"], rollouts=s["rollouts"], table=table, failures=failures), None
    raise ValueError(step)


# ------------------------------------------------------------------ summary.json
def summary(state):
    turns = []
    for t in state["turns"]:
        st = t["steps"]
        row = dict(turn=t["turn"], steps={n: dict(state=s["state"], attempts=len(s["attempts"]),
                                                  wall_s=sum(a.get("wall_s") or 0 for a in s["attempts"]) or None,
                                                  error=(s["attempts"][-1].get("error") if s["attempts"] else None),
                                                  reason=s.get("reason"))
                                          for n, s in st.items()})
        for n in ("eval_heldout", "eval_probe"):
            o = st[n]["outputs"]
            if o:
                row[n.split("_", 1)[1]] = {k: o[k] for k in ("all_pass", "first_try_passed", "rollouts", "table")}
        row["data"] = {k: st["dataset"]["outputs"].get(k) for k in ("dataset", "episodes", "new_episodes")}
        row["train"] = {k: st["train"]["outputs"].get(k) for k in ("ckpt", "steps", "final_loss")}
        turns.append(row)
    return dict(exp=state["exp"], status=state["status"], stop_reason=state["stop_reason"], error=state["error"],
                turns=turns, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def save(state, paths):
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    J.write_json(paths.exp / "state.json", state)
    J.write_json(paths.exp / "summary.json", summary(state))
