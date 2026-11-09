"""Recovery demonstrations (host side, or the manager's container): per failed rollout, the failed phase
regenerated from the rollout's saved phase start state with AutoMoMa, until episodes_per_failure
suffix episodes are accepted. Ported from the earlier pipeline's regenerate stage:
  1. plan job (revolve-recovery, GPU; src/recovery/plan_from_state.py): the rest of the task from the
     saved state. An open restart plans the pull from the grasped state (AKR trajopt; a policy rollout's
     on the grasp it actually holds, grasp_from_state); an approach or grasp restart plans a MotionGen
     reach to a new grasp plus its pull. Planner runs of `attempts` problems, seeds seed0 + runs*i + r;
  2. per planned attempt a camera-off check (revolve-simulator, src/recovery/regen_sim.py candidate):
     the regenerated part's actions, the clearance precheck, the gz run restored from the saved state
     (task.yaml phases from the restarted one on + the offline verdict of the regenerated part);
  3. an accepted check is recorded with the training cameras (the same state log required) into an
     MCAP bag; the first accepted recording is episode i. Episodes are SUFFIX-ONLY: frame 0 = the
     restored mid-task state; episode.json "recovery" says where it came from.

  python components/recovery/recover.py --task T --rollouts R1[:PHASE] R2 ... --out runs/<exp>/turn_<k>/recovery \
      [--episodes-per-failure 3] [--phases approach,grasp,open] [--workers 2]
  recover(task, rollouts, out, episodes_per_failure=3, phases=..., workers=2, jobs=DOCKER) -> manifest

A rollout dir (what evaluation hands over): phase_states/<phase>_start.npz (+ _next), phase_report.json
(the simulator's per-phase verdict: result.failed_phase, result.regenerate), episode.json (task, and
seed / policy_rollout / plan_run, attempt_id, grasp_id when known). `R:PHASE` forces the phase (tests,
an accepted demo's states). Release and untimed global failures are not regenerated ("unsupported").

Layout under out/:
  manifest.json                       rewritten atomically after every job; a re-run resumes from it
  <rollout>/plans/e<i>_r<r>_s<seed>/  plan_from_state run (regen.json, start_state*.npz, handoff.*, assets/)
  <rollout>/checks/<plan>_a<a>/       camera-off candidate run (precheck.json, verdict.json, phase_report.json, ...)
  <rollout>/episodes/ep_<i>/          the recorded episode: bag/, bag_check.json, verdict.json, phase_report.json,
                                      phase_states/, state_log.npz, episode.json (+ "recovery" block)
  <rollout>/jobs/<kind>_<name>/       each container job's job.json, status.json, result.json, logs
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "components" / "datagen")]
import generate as G  # noqa: E402  (datagen: read, sha256, judge, specs_bridge on sys.path)
import specs_bridge  # noqa: E402

from revolve_common import job as J  # noqa: E402

IMAGE = os.environ.get("REVOLVE_RECOVERY_IMAGE", "revolve-recovery:latest")
ENTRY = "/repo/components/recovery/src/recovery/recovery_entry.py"
REGEN_SIM = "/repo/components/recovery/src/recovery/regen_sim.py"
REGENERABLE = {"approach": "reach", "grasp": "reach", "open": "pull"}
# the phase the restored run is judged from: a grasp restart re-approaches (reach, approach in, close)
RESTORE_PHASE = {"approach": "approach", "grasp": "approach", "open": "open"}
RENAMED = {"fridge_door": "open_fridge_door", "drawer_open": "open_drawer"}
PLAN_TIMEOUT_S = 1200
CHECK_TIMEOUT_S = 600
RECORD_TIMEOUT_S = 900
# planners on the GPU at once across all threads of this process (8 GB laptop: 2 fit, 3 OOM)
PLANNERS = threading.Semaphore(int(os.environ.get("REVOLVE_GPU_PLANNERS", "2")))


# ------------------------------------------------------------------ jobs (docker by default; stubs in tests)
_LOAD = threading.Lock()


def _datagen_launch():
    with _LOAD:          # registered before exec (its dataclass needs it): other threads wait for the exec
        if "datagen_launch" not in sys.modules:
            s = importlib.util.spec_from_file_location("datagen_launch", REPO / "components" / "datagen" / "launch.py")
            sys.modules["datagen_launch"] = m = importlib.util.module_from_spec(s)
            s.loader.exec_module(m)
        return sys.modules["datagen_launch"]


def start_plan(args, job_dir, **kw):
    """A recovery_entry.py job in revolve-recovery (datagen's launcher: mounts, GPU, wrapper)."""
    return _datagen_launch().start(args, job_dir, image=IMAGE, entry=ENTRY, component="recovery",
                                   parser="recovery_progress:parse", pythonpath=["/repo/components/recovery/src/recovery"], **kw)


def start_sim(args, job_dir, **kw):
    """A regen_sim.py job in revolve-simulator (the simulator's launcher)."""
    return _datagen_launch().sim_launch.start(args, job_dir, cli=REGEN_SIM, **kw)


def finish(job):
    return _datagen_launch().finish(job)


@dataclass(frozen=True)
class Jobs:
    plan: Callable[..., int | None]        # (plan_dir, seed, attempts, job_dir) -> exit code
    candidate: Callable[..., int | None]   # (plan_dir, attempt, out, job_dir, cameras) -> exit code


def docker_plan(plan_dir, seed, attempts, job_dir):
    with PLANNERS:
        return finish(start_plan(["plan", "--dir", str(plan_dir), "--seed", str(seed), "--attempts", str(attempts)],
                                 job_dir, timeout=PLAN_TIMEOUT_S))["exit"]


def docker_candidate(plan_dir, attempt, out, job_dir, cameras=0):
    args = ["candidate", "--plan", str(plan_dir), "--attempt", str(attempt), "--out", str(out)]
    args += ["--cameras", str(cameras)] if cameras else ["--no-bag"]
    return finish(start_sim(args, job_dir, gpu=bool(cameras),
                            timeout=RECORD_TIMEOUT_S if cameras else CHECK_TIMEOUT_S))["exit"]


DOCKER = Jobs(docker_plan, docker_candidate)


# ------------------------------------------------------------------ the rollout's failure
def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def source(d, task, phase=None):
    """What a rollout dir says about its failure (see the module doc)."""
    d = Path(d).resolve()
    res = (G.read(d / "phase_report.json") or {}).get("result") or {}
    ep = G.read(d / "episode.json") or {}
    live = G.read(d / "live.json") or {}
    t = RENAMED.get(ep.get("task"), ep.get("task"))
    if t and t != task:
        raise ValueError(f"{d}: task {t}, not {task}")
    seed = ep.get("seed", (ep.get("datagen") or {}).get("seed", (ep.get("rollout") or {}).get("seed")))
    return dict(dir=rel(d), failed_phase=res.get("failed_phase"), phase=phase or res.get("failed_phase"),
                forced=phase is not None, regenerate=res.get("regenerate"), failed_checks=res.get("failed_checks", []),
                policy="policy_rollout" in ep or bool(live.get("policy")), seed=seed,
                plan_run=ep.get("plan_run"), attempt_id=ep.get("attempt_id"), grasp_id=ep.get("grasp_id"))


def key(d, phase=None):
    """A rollout's name in the manifest and out/: its dir name (+ _from_<phase> when the phase is forced)."""
    return Path(d).name + (f"_from_{phase}" if phase else "")


def unsupported(src, phases):
    """None if the failure is regenerated, else (state, reason)."""
    p = src["phase"]
    if p is None:
        return "passed", "every phase passed: nothing to recover"
    if p not in REGENERABLE:
        return "unsupported", f"failed phase {p} ({src['failed_checks']}): only {sorted(REGENERABLE)} are regenerated"
    if not src["forced"] and src["regenerate"] is False:
        return "unsupported", f"failure not regenerable ({src['failed_checks']})"
    if p not in phases:
        return "skipped", f"phase {p} not in the recovery phases {phases}"
    if not (REPO / src["dir"] / "phase_states" / f"{p}_start.npz").is_file():
        return "error", f"no phase_states/{p}_start.npz"
    return None


# ------------------------------------------------------------------ manifest
class Manifest:
    def __init__(self, out, params):
        self.path, self.lock = Path(out) / "manifest.json", threading.Lock()
        self.m = G.read(self.path) or dict(params, rollouts={}, created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if self.m.get("task") != params["task"]:
            raise SystemExit(f"{self.path}: task {self.m.get('task')!r}, not {params['task']!r}")
        self.m.update(params)

    def save(self):
        m = self.m
        recs = [m["rollouts"][n] for n in m["rollout_list"]]
        m["episodes"] = sorted(e["dir"] for r in recs for e in r.get("episodes", []) if e["accepted"])
        m["counts"] = dict(rollouts=len(recs), rollouts_done=sum(r["state"] not in ("pending", "running") for r in recs),
                           unsupported=sum(r["state"] in ("unsupported", "skipped", "passed") for r in recs),
                           plans=sum(len(r.get("plans", {})) for r in recs),
                           candidates_checked=sum(len(r.get("checks", {})) for r in recs),
                           episodes_accepted=len(m["episodes"]),
                           episodes_target=sum(m["episodes_per_failure"] for r in recs if r.get("mode")))
        m["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        J.write_json(self.path, m)
        print("PROGRESS " + json.dumps(m["counts"]), flush=True)


# ------------------------------------------------------------------ recovery of one rollout
class Recovery:
    def __init__(self, task, rollouts, out, episodes_per_failure=3, phases=tuple(REGENERABLE), workers=2,
                 jobs=DOCKER, say=lambda msg: print(msg, flush=True)):
        self.task, self.out = task, Path(out).resolve()
        self.items = [(Path(d).resolve(), p) for d, p in (x if isinstance(x, tuple) else (x, None) for x in rollouts)]
        names = [key(d, p) for d, p in self.items]
        if len(set(names)) != len(names):
            raise SystemExit(f"rollout names must be unique: {names}")
        self.eps, self.phases, self.workers, self.jobs, self._say = episodes_per_failure, list(phases), workers, jobs, say
        R = specs_bridge.generator()["regenerate"]
        self.runs, self.attempts, self.seed0 = R["runs"], R["attempts"], R["seed0"]
        self.fps = int(specs_bridge.task_spec(task).robot.cameras[0].rate)
        self.cameras = [c.name for c in specs_bridge.task_spec(task).robot.cameras]
        self.man = Manifest(self.out, dict(task=task, rollout_list=names, episodes_per_failure=episodes_per_failure,
                                           phases=self.phases, runs=self.runs, attempts=self.attempts, seed0=self.seed0))

    def say(self, msg):
        self._say(f"{time.strftime('%H:%M:%S')} recover {self.task}: {msg}")

    def rel(self, p):
        return str(Path(p).relative_to(self.out))

    def update(self, fn):
        with self.man.lock:
            fn()
            self.man.save()

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.update(lambda: [self.man.m["rollouts"].setdefault(key(d, p), dict(state="pending")) for d, p in self.items])
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(ex.map(self.rollout_safe, self.items))
        self.say(f"done: {self.man.m['counts']}")
        return self.man.m

    def rollout_safe(self, item):
        name = key(*item)
        rec = self.man.m["rollouts"][name]
        try:
            self.rollout(name, *item)
        except Exception as e:         # an infrastructure error ends this rollout, not the others
            self.update(lambda: rec.update(state="error", reason=repr(e)))
            self.say(f"{name}: error {e!r}")

    def rollout(self, name, d, force_phase):
        rec = self.man.m["rollouts"][name]
        src = source(d, self.task, force_phase)
        bad = unsupported(src, self.phases)
        if bad:
            self.update(lambda: rec.update(source=src, state=bad[0], reason=bad[1]))
            self.say(f"{name}: {bad[0]}: {bad[1]}")
            return
        phase = src["phase"]
        snap = d / "phase_states" / f"{phase}_start.npz"
        grasp_from_state = phase == "open" and (src["policy"] or src["grasp_id"] is None)
        def begin():
            for k in ("plans", "checks"):
                rec.setdefault(k, {})
            rec.setdefault("episodes", [])
            rec.update(source=src, state="running", phase=phase, mode=REGENERABLE[phase],
                       restore_phase=RESTORE_PHASE[phase], grasp_from_state=grasp_from_state,
                       action_index=int(np.load(snap)["action_index"]))
        self.update(begin)
        self.say(f"{name}: recover {phase} ({REGENERABLE[phase]}) from {rel(snap)}, source action {rec['action_index']}"
                 + (", grasp from the saved state" if grasp_from_state else ""))
        for i in range(self.eps):
            if any(e["index"] == i and e["accepted"] for e in rec["episodes"]):
                continue
            e = self.episode(name, d, rec, i)
            self.say(f"{name}: episode {i} " + (f"accepted ({e['plan']} a{e['attempt_id']})" if e else
                                                f"not found in {self.runs} planner runs x {self.attempts} attempts"))
        n = sum(e["accepted"] for e in rec["episodes"])
        self.update(lambda: rec.update(state="done" if n >= self.eps else "short",
                                       reason=None if n >= self.eps else f"{n} of {self.eps} accepted, budget used up"))

    def episode(self, name, d, rec, i):
        rdir = self.out / name
        for r in range(self.runs):
            seed = self.seed0 + self.runs * i + r
            pkey = f"e{i}_r{r}_s{seed}"
            pdir = rdir / "plans" / pkey
            p = rec["plans"].get(pkey)
            if p is None or p["rc"] != 0:                 # a crashed plan run is redone on a resume
                p = self.plan(name, d, rec, pdir, seed, r)
                self.update(lambda: rec["plans"].__setitem__(pkey, p))
                self.say(f"{name} {pkey}: {len(p['planned'])} of {p['attempts']} planned" +
                         ("" if p["rc"] == 0 else f", plan exit {p['rc']}"))
            for a in p["planned"]:
                ckey = f"{pkey}_a{a}"
                c = rec["checks"].get(ckey)
                if c is None:
                    cdir = rdir / "checks" / ckey
                    rc = self.jobs.candidate(pdir, a, cdir, rdir / "jobs" / f"check_{ckey}", 0)
                    c = dict(G.judge(cdir, rc), dir=self.rel(cdir), precheck=rc != 2)
                    self.update(lambda: rec["checks"].__setitem__(ckey, c))
                    self.say(f"{name} {ckey}: " + ("precheck failed" if rc == 2 else "accepted" if c["accepted"] else
                                                      f"rejected ({c['failed_phase']}; {'; '.join(c['reasons'])[:200]})"))
                if not c["accepted"]:
                    continue
                if any(e["check"] == c["dir"] for e in rec["episodes"]):
                    continue                              # recorded before and rejected
                e = self.record(name, rec, i, pkey, pdir, a, c)
                self.update(lambda: rec["episodes"].append(e))
                if e["accepted"]:
                    return e
                self.say(f"{name} ep_{i} from {ckey}: record rejected: {e['reasons']}")
        return None

    def plan(self, name, d, rec, pdir, seed, r):
        shutil.rmtree(pdir, ignore_errors=True)
        pdir.mkdir(parents=True)
        src, phase = rec["source"], rec["phase"]
        for s in ("", "_next"):
            shutil.copy(d / "phase_states" / f"{phase}_start{s}.npz", pdir / f"start_state{s}.npz")
        J.write_json(pdir / "regen.json", dict(
            task=self.task, phase=phase, mode=rec["mode"], restore_phase=rec["restore_phase"], source_rollout=src["dir"],
            source_label=f"{self.task}/{name}", source_plan_run=src["plan_run"], source_attempt_id=src["attempt_id"],
            source_grasp_id=src["grasp_id"], action_index=rec["action_index"], planner_run=r,
            **({"grasp_from_state": True} if rec["grasp_from_state"] else {})))
        rc = self.jobs.plan(pdir, seed, self.attempts, self.out / name / "jobs" / f"plan_{pdir.name}")
        h = G.read(pdir / "handoff.json") if rc == 0 else None
        return dict(rc=rc if h else (rc or 3), dir=self.rel(pdir), seed=seed, run=r, attempts=(h or {}).get("attempts", 0),
                    planned=[x["attempt_id"] for x in (h or {}).get("attempts_detail", []) if x["success"]])

    def record(self, name, rec, i, pkey, pdir, a, c):
        edir = self.out / name / "episodes" / f"ep_{i}"
        rc = self.jobs.candidate(pdir, a, edir, self.out / name / "jobs" / f"record_ep_{i}_{pkey}_a{a}", self.fps)
        v, bc = G.judge(edir, rc), G.read(edir / "bag_check.json") or {}
        reasons = ([] if v["accepted"] else [f"exit {rc}: {v['failed_phase']} {v['reasons']}"]) + \
            ([] if v["state_log_sha256"] == c["state_log_sha256"] else ["state_log.npz differs from the check's"]) + \
            ([] if bc.get("pass") is True else [f"bag check failed: {bc.get('failed')}"])
        for cam in self.cameras:               # the bag holds the frames as JPEG
            shutil.rmtree(edir / cam, ignore_errors=True)
        for f in (edir / "restore").glob("*.npz"):  # the restored start: this episode's first phase state
            if (edir / "phase_states").is_dir() and not (edir / "phase_states" / f.name).exists():
                shutil.copy2(f, edir / "phase_states" / f.name)
        src = rec["source"]
        e = dict(index=i, name=f"ep_{i}", dir=self.rel(edir), plan=pkey, attempt_id=a, check=c["dir"], accepted=not reasons,
                 reasons=reasons, state_log_sha256=v["state_log_sha256"])
        ep = G.read(edir / "episode.json")
        if ep is not None:
            ep["recovery"] = dict(source="recovery", label="recovery", phase=rec["phase"], suffix=True,
                                  from_rollout=src["dir"], rollout=name, rollout_seed=src["seed"], task=self.task,
                                  failed_checks=src["failed_checks"], mode=rec["mode"], restore_phase=rec["restore_phase"],
                                  grasp_from_state=rec["grasp_from_state"], action_index_offset=rec["action_index"],
                                  plan=f"plans/{pkey}", planner_seed=rec["plans"][pkey]["seed"], attempt_id=a,
                                  check=c["dir"], accepted=e["accepted"])
            J.write_json(edir / "episode.json", ep)
        return e


def recover(task, rollouts, out, episodes_per_failure=3, phases=tuple(REGENERABLE), workers=2, jobs=DOCKER, say=None):
    kw = {} if say is None else dict(say=say)
    return Recovery(task, rollouts, out, episodes_per_failure, phases, workers, jobs, **kw).run()


def parse_rollout(text):
    d, _, p = text.rpartition(":") if ":" in Path(text).name else (text, "", "")
    return (d, p) if p else d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--rollouts", required=True, nargs="+", help="rollout dirs, each optionally :PHASE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes-per-failure", type=int, default=3)
    ap.add_argument("--phases", default=",".join(REGENERABLE), help="phases recovered (experiment recovery.phases)")
    ap.add_argument("--workers", type=int, default=2, help="rollouts at a time (planners capped by REVOLVE_GPU_PLANNERS)")
    a = ap.parse_args(argv)
    m = recover(a.task, [parse_rollout(x) for x in a.rollouts], a.out, a.episodes_per_failure, a.phases.split(","), a.workers)
    return 1 if any(r["state"] == "error" for r in m["rollouts"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
