"""Start states of the evaluation seeds (host side). Ported from the legacy held-out start design: an eval round
on unused seeds run `--until mint` (the held-out eval round), then its starts script took
each accepted episode's spawn. Here, per seed:

  seed S -> the datagen plan run for S (components/datagen: AutoMoMa planner seed S, clustered starts centred
            by S, start_design.py) -> its planned attempts replay-checked in order (camera off) -> the FIRST
            accepted check is the start: its spawn (actions.npz row 0: base x, y, yaw, the arm joints, gripper), the
            planned grasp (live.json grasp_pose_world, the approach gate's reference) and the plan ids.

So a train seed's start is its demo's start, and a probe / held-out seed's start is what a demo of it would
have started from. The record is written once to <out>/seed_<S>/start.json and reused: the same seed gives
the same start for the whole experiment even where cuRobo plans are not repeatable (the drawer; fridge plans
are bit-identical run to run). No novelty check vs the training starts (starts.py did that for
hand-picked eval rounds; here the seed sets are disjoint by construction, specs.Seeds).

  resolve(task, seeds, out, sources=(), workers=2, jobs=G.DOCKER) -> {seed: start record}
  python components/evaluation/starts.py --task T --seeds 100-103 --out runs/<exp>/starts [--source DATAGEN_OUT ...]

sources: datagen output dirs (generate.py: manifest.json + checks/) searched first, e.g. turn 0's data for
train seeds or a test cache; else plan + check jobs run into <out>/plans/seed_<S>/, <out>/checks/seed_<S>_a<A>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "components" / "datagen")]
import generate as G  # noqa: E402  (datagen: docker_plan, docker_replay, judge, read, parse_seeds)

from revolve_common import job as J  # noqa: E402

_LOCK = threading.Lock()


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def record(task, seed, check_dir, source):
    """start.json of an accepted check dir (episode.json, actions.npz, live.json, state_log.npz)."""
    d = Path(check_dir)
    ep, live = G.read(d / "episode.json"), G.read(d / "live.json")
    a = np.load(d / "actions.npz")
    return dict(task=task, seed=int(seed), dir=rel(d), source=source, plan_run=ep["plan_run"], attempt_id=ep["attempt_id"],
                grasp_id=ep["grasp_id"], spawn=dict(base=a["base"][0].tolist(), arm=a["arm"][0].tolist(),
                                                    grip=float(a["grip"][0])),
                grasp_pose_world=live["grasp_pose_world"], actions_sha256=sha256(d / "actions.npz"),
                state_log_sha256=sha256(d / "state_log.npz"))


def from_sources(task, seed, sources):
    """The first accepted check of `seed` in a datagen output dir (its manifest's planned order)."""
    for src in sources:
        m = G.read(Path(src) / "manifest.json") or {}
        if m.get("task") != task:
            continue
        s = (m.get("seeds") or {}).get(str(seed)) or {}
        for a in (s.get("plan") or {}).get("planned", []):
            c = (s.get("checks") or {}).get(str(a))
            if c and c.get("accepted") and (Path(src) / c["dir"] / "actions.npz").is_file():
                return Path(src) / c["dir"]
    return None


def resolve_one(task, seed, out, sources=(), jobs=G.DOCKER, say=print):
    out = Path(out)
    f = out / f"seed_{seed}" / "start.json"
    rec = G.read(f)
    if rec is not None:
        if rec["task"] != task:
            raise ValueError(f"{f}: task {rec['task']}, not {task}")
        if sha256(REPO / rec["dir"] / "actions.npz") != rec["actions_sha256"]:
            raise ValueError(f"{f}: {rec['dir']}/actions.npz changed since the start was resolved")
        return rec
    d = from_sources(task, seed, sources)
    source = "source"
    if d is None:
        source = "planned"
        plan = out / "plans" / f"seed_{seed}"
        say(f"start seed {seed}: plan")
        rc = jobs.plan(task, seed, plan, out / "jobs" / f"plan_seed_{seed}", None)
        h = G.read(plan / "handoff.json") or {}
        planned = [r["attempt_id"] for r in h.get("attempts_detail", []) if r["success"]] if rc == 0 else []
        for a in planned:
            c = out / "checks" / f"seed_{seed}_a{a}"
            ok = G.judge(c, jobs.replay(task, plan, a, c, out / "jobs" / f"check_seed_{seed}_a{a}", 0))["accepted"]
            say(f"start seed {seed}: attempt {a} {'accepted' if ok else 'rejected'}")
            if ok:
                d = c
                break
        if d is None:
            raise ValueError(f"seed {seed}: no accepted attempt (plan exit {rc}, {len(planned)} planned)")
    rec = record(task, seed, d, source)
    with _LOCK:
        J.write_json(f, rec)
    return rec


def resolve(task, seeds, out, sources=(), workers=2, jobs=G.DOCKER, say=print):
    """{seed: start record}; seeds whose start cannot be made raise (the experiment cannot evaluate them)."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        recs = list(ex.map(lambda s: resolve_one(task, s, out, sources, jobs, say), seeds))
    return dict(zip(seeds, recs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", required=True, help="0-9 / 0,3,5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", nargs="*", default=[], help="datagen output dirs to take accepted checks from first")
    ap.add_argument("--workers", type=int, default=2, help="seeds planned at once (GPU planners)")
    a = ap.parse_args(argv)
    recs = resolve(a.task, G.parse_seeds(a.seeds), a.out, a.source, a.workers)
    for s, r in recs.items():
        print(f"seed {s}: {r['dir']} ({r['source']}), base {np.round(r['spawn']['base'], 3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
