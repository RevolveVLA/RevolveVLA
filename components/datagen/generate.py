"""Episode generation (host side, or the manager's container): per seed, an AutoMoMa plan run (datagen
job, GPU), then the planned attempts in order through the simulator: a camera-off replay check, and for
an accepted check the camera replay that records the episode (MCAP bag), until the seed has
episodes_per_seed accepted episodes or its planned attempts run out. `workers` seeds at a time.

  python components/datagen/generate.py --task T --seeds 0-9 --out runs/<exp>/turn_0/data \
      [--episodes-per-seed 1] [--workers 2] [--attempts N] [--variant NAME] [--checks 1]
  generate(task, seeds, out, episodes_per_seed=1, workers=2, attempts=None, jobs=DOCKER, say=None, layouts=None,
           checks=1, record=True) -> manifest

--checks N (N > 1): an attempt passes its check only when N camera-off replays of the same plan from the
same start are all accepted (repeat_check: repeats 1..N-1 run into checks/seed_<s>_a<a>_r<k>/, stopping
at the first rejection). The check record then adds repeats and repeat_check {n, accepted, state_log_sha256
(one per repeat run), identical, reason "repeat check k/N failed"}, and its `accepted` is the N-of-N
verdict; the manifest adds checks. A resume continues from the repeats already recorded.

layouts {seed: layout} (start_design.parse_layout: start_pose and / or grasp_id, what the datagen plugin
realised): the seed's plan run starts from it (entry.py --layout) and episode.json "datagen" adds the layout.
record=False stops a seed at its first accepted check (the "episodes" are those checks, label "check": no camera
replay, no bag): a split whose episodes are not recorded, e.g. held-out start states (datagen.plugins.automoma).

Layout under out/ (ported from the earlier pipeline's plans/ check/ record/ stages):
  manifest.json              rewritten atomically after every plan, check and record; a re-run resumes
                             from it (finished items are skipped, a half-written item is redone)
  plans/seed_<s>/            the plan run (handoff.npz/json, plan_report.json, fk_check.json, assets/)
  checks/seed_<s>_a<a>/      the attempt's camera-off replay (verdict.json, phase_report.json, state_log.npz)
  checks/seed_<s>_a<a>_r<k>/ its repeat k (--checks > 1)
  episodes/seed_<s>_a<a>/    the camera replay: bag/, bag_check.json, verdict.json, phase_report.json,
                             phase_states/, state_log.npz, episode.json (+ "datagen": seed, attempt, label),
                             plan.json + plan.npz (the attempt's planned rows); the camera PNGs are removed
                             once the bag (JPEG) is checked, as the legacy record stage did
  jobs/<kind>_<name>/        each container job's job.json, status.json, result.json, stdout.log, container.log
An episode is accepted when its check passed (replay exit 0: the offline verdict accepted and every
task.yaml phase passed), its camera replay passed too, reproduced the check's state_log.npz (sha256) and
its bag check passed. PROGRESS {json} lines (seeds done, episodes planned / checked / accepted) feed the
wrapper's /status when this runs under revolve_common.wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
sys.path[:0] = [str(HERE), str(HERE / "src" / "datagen")]
import specs_bridge  # noqa: E402

from revolve_common import job as J  # noqa: E402

PLAN_TIMEOUT_S = 1800
CHECK_TIMEOUT_S = 600            # a camera-off replay takes 30-35 s
RECORD_TIMEOUT_S = 900           # a camera replay takes 36-50 s


# ------------------------------------------------------------------ jobs (docker by default; stubs in tests)
@dataclass(frozen=True)
class Jobs:
    plan: Callable[..., int | None]       # (task, seed, out, job_dir, attempts[, layout=]) -> exit code
    replay: Callable[..., int | None]     # (task, plan_dir, attempt, out, job_dir, cameras_hz) -> exit code


def docker_plan(task, seed, out, job_dir, attempts=None, layout=None):
    import launch
    args = ["plan", "--task", task, "--seed", str(seed), "--out", str(out)]
    args += ["--layout", json.dumps(layout)] if layout is not None else []
    return launch.run(args + (["--attempts", str(attempts)] if attempts else []), job_dir, timeout=PLAN_TIMEOUT_S)["exit"]


def docker_replay(task, plan_dir, attempt, out, job_dir, cameras=0):
    import launch
    env = {f"ROUND_{k.upper()}": json.dumps(v) for k, v in specs_bridge.round_blocks(task).items()
           if k in ("timing", "pull")}
    args = ["replay", "--plan", str(plan_dir), "--attempt", str(attempt), "--task", task, "--out", str(out)]
    args += ["--cameras", str(cameras)] if cameras else ["--no-bag"]
    return launch.sim_launch.run(args, job_dir, gpu=bool(cameras), env=env,
                                 timeout=RECORD_TIMEOUT_S if cameras else CHECK_TIMEOUT_S)["exit"]


DOCKER = Jobs(docker_plan, docker_replay)


# ------------------------------------------------------------------ helpers
def read(p):
    p = Path(p)
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except ValueError:
        return None


def sha256(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def parse_seeds(text):
    """'0-9' / '0,3,5' / '0-3,8' -> [ints]."""
    out = []
    for part in str(text).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def judge(d, rc):
    """A replay dir's verdict: exit 0 = offline verdict accepted and every phase passed."""
    v, pr = read(Path(d) / "verdict.json") or {}, read(Path(d) / "phase_report.json") or {}
    return dict(rc=rc, accepted=bool(rc == 0 and v.get("verdict") == "accepted" and pr.get("accepted")),
                verdict=v.get("verdict"), reasons=v.get("reasons", []), failed_phase=pr.get("result"),
                state_log_sha256=sha256(Path(d) / "state_log.npz"))


def repeat_check(n, chk, run, save=lambda chk: None):
    """The N-of-N check rule on one attempt -> chk with repeats and repeat_check. chk: repeat 0's check (judge +
    dir); run(k) -> repeat k's (camera off, its own dirs). Repeats stop at the first rejection and continue
    from chk's repeat_check / repeats (a resume); save(new chk) after each (fresh dicts: chk is not mutated,
    the manifest may be serialising it). n == 1 or done: chk."""
    if n == 1 or repeat_done(chk, n):
        return chk
    rc = chk.get("repeat_check")
    acc, shas = ((list(rc["accepted"]), list(rc["state_log_sha256"])) if rc
                 else ([chk["accepted"]], [chk["state_log_sha256"]]))
    reps = list(chk.get("repeats", []))
    while all(acc) and len(acc) < n:
        save(dict(chk, accepted=False, repeats=list(reps), repeat_check=dict(n=n, accepted=list(acc),
                                                                            state_log_sha256=list(shas))))
        r = run(len(acc))
        reps.append(r)
        acc.append(r["accepted"])
        shas.append(r["state_log_sha256"])
    bad = next((k for k, ok in enumerate(acc) if not ok), None)
    chk = dict(chk, accepted=bad is None, repeats=reps, repeat_check=dict(
        n=n, accepted=acc, state_log_sha256=shas, identical=None not in shas and len(set(shas)) == 1,
        reason=None if bad is None else f"repeat check {bad + 1}/{n} failed"))
    save(chk)
    return chk


def repeat_done(chk, n):
    rc = chk.get("repeat_check") or {}
    return rc.get("n") == n and "reason" in rc


def plan_failure(rc, planned, handoff, ok):
    """Why a plan run left nothing to check (plan failed / no IK / FK check failed), or None. handoff: a
    callable -> handoff.json (read only when nothing was planned)."""
    if rc not in (0, 1):
        return "plan failed"
    if not planned:
        rows = (handoff() or {}).get("attempts_detail", [])
        ik = any(r.get("goal_ik") == "seeded" or r.get("goal_ik_cluster", -1) >= 0 for r in rows)
        return "plan failed" if ik else "no IK"
    return None if ok else "FK check failed"


class Manifest:
    """manifest.json: one writer (any thread, under the lock), atomic writes."""

    def __init__(self, out, params):
        self.path, self.lock = Path(out) / "manifest.json", threading.Lock()
        self.m = read(self.path) or dict(params, seeds={}, created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if self.m.get("task") != params["task"]:
            raise SystemExit(f"{self.path}: task {self.m.get('task')!r}, not {params['task']!r}")
        self.m.update(params)

    def seed(self, s):
        return self.m["seeds"].setdefault(str(s), dict(state="pending", plan=None, checks={}, episodes=[]))

    def save(self):
        m = self.m
        recs = [r for k, r in m["seeds"].items() if int(k) in m["seed_list"]]
        m["episodes"] = sorted(e["name"] for r in recs for e in r["episodes"] if e["accepted"])
        m["counts"] = dict(seeds=len(m["seed_list"]), seeds_done=sum(r["state"] in ("done", "short", "failed") for r in recs),
                           episodes_planned=sum(len((r["plan"] or {}).get("planned", [])) for r in recs),
                           episodes_checked=sum(len(r["checks"]) for r in recs),
                           episodes_accepted=len(m["episodes"]),
                           episodes_target=len(m["seed_list"]) * m["episodes_per_seed"])
        m["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        J.write_json(self.path, m)
        print("PROGRESS " + json.dumps(m["counts"]), flush=True)


# ------------------------------------------------------------------ one seed
class Generation:
    def __init__(self, task, seeds, out, episodes_per_seed=1, workers=2, attempts=None, jobs=DOCKER,
                 say=lambda msg: print(msg, flush=True), layouts=None, checks=1, record=True):
        self.task, self.seeds, self.out = task, list(seeds), Path(out).resolve()
        self.eps, self.workers, self.attempts, self.jobs, self._say = episodes_per_seed, workers, attempts, jobs, say
        self.layouts, self.checks, self.recording = layouts or {}, checks, record
        robot = specs_bridge.task_spec(task).robot
        self.cameras = [c.name for c in robot.cameras]
        self.fps = int(robot.cameras[0].rate)
        self.man = Manifest(self.out, dict(task=task, seed_list=self.seeds, episodes_per_seed=episodes_per_seed,
                                           attempts=attempts or specs_bridge.load("generator", "automoma")["attempts"],
                                           **(dict(checks=checks) if checks > 1 else {})))

    def say(self, msg):
        self._say(f"{time.strftime('%H:%M:%S')} generate {self.task}: {msg}")

    def rel(self, p):
        return str(Path(p).relative_to(self.out))

    def update(self, fn):
        with self.man.lock:
            fn()
            self.man.save()

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.update(lambda: [self.man.seed(s) for s in self.seeds])
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(ex.map(self.seed_safe, self.seeds))
        self.say(f"done: {self.man.m['counts']}")
        return self.man.m

    def seed_safe(self, s):
        try:
            self.seed(s)
        except Exception as e:         # an infrastructure error ends this seed, not the others
            rec, reason = self.man.seed(s), repr(e)
            self.update(lambda: rec.update(state="error", reason=reason))
            self.say(f"seed {s}: error {e!r}")

    def seed(self, s):
        rec = self.man.seed(s)
        plan_dir = self.out / "plans" / f"seed_{s}"
        p = rec["plan"]
        if p is None or not (p["ok"] or p["rc"] == 1):       # rc 1: the FK check failed (kept); else redo
            self.say(f"seed {s}: plan")
            rc = self.jobs.plan(self.task, s, plan_dir, self.out / "jobs" / f"plan_seed_{s}", self.attempts,
                                **({"layout": self.layouts[s]} if s in self.layouts else {}))
            h, fk = read(plan_dir / "handoff.json"), read(plan_dir / "fk_check.json")
            ok = bool(rc == 0 and h and fk and fk.get("passed"))
            p = dict(ok=ok, rc=rc, dir=self.rel(plan_dir), attempts=(h or {}).get("attempts"),
                     planned=[r["attempt_id"] for r in (h or {}).get("attempts_detail", []) if r["success"]],
                     fk_passed=(fk or {}).get("passed"))
            self.update(lambda: rec.update(plan=p, state="planned" if ok else "failed",
                                           reason=None if ok else f"plan exit {rc}"))
            self.say(f"seed {s}: {len(p['planned'])} of {p['attempts']} planned" + ("" if ok else f", plan exit {rc}"))
        if not p["ok"]:
            return
        for a in p["planned"]:
            if sum(e["accepted"] for e in rec["episodes"]) >= self.eps:
                break
            if any(e["attempt_id"] == a for e in rec["episodes"]):
                continue
            chk = rec["checks"].get(str(a))
            if chk is None:
                chk = self.check(s, a, plan_dir)
                self.update(lambda: rec["checks"].__setitem__(str(a), chk))
                self.say(f"seed {s} a{a}: check " + ("accepted" if chk["accepted"] else
                                                       f"rejected ({chk['failed_phase']}; {'; '.join(chk['reasons'])[:200]})"))
            if self.checks > 1 and not repeat_done(chk, self.checks):
                chk = repeat_check(self.checks, chk, lambda k: self.check(s, a, plan_dir, k),
                                   lambda c: self.update(lambda: rec["checks"].__setitem__(str(a), c)))
                r = chk["repeat_check"]
                self.say(f"seed {s} a{a}: " + (r["reason"] or f"{self.checks}/{self.checks} checks accepted")
                         + f", state logs {'identical' if r['identical'] else 'differ'}")
            if chk["accepted"]:
                e = self.record(s, a, plan_dir, chk) if self.recording else self.checked(s, a, chk)
                self.update(lambda: rec["episodes"].append(e))
                self.say(f"seed {s} a{a}: episode {e['name']} " + ("accepted" if e["accepted"] else f"rejected: {e['reasons']}"))
        n = sum(e["accepted"] for e in rec["episodes"])
        self.update(lambda: rec.update(state="done" if n >= self.eps else "short",
                                       reason=None if n >= self.eps else f"{n} of {self.eps} accepted, planned attempts used up"))

    def check(self, s, a, plan_dir, k=0):
        """Camera-off replay k of an attempt (k 0: the check; k >= 1: its repeats) -> judge + dir."""
        name = f"seed_{s}_a{a}" + (f"_r{k}" if k else "")
        cdir = self.out / "checks" / name
        rc = self.jobs.replay(self.task, plan_dir, a, cdir, self.out / "jobs" / f"check_{name}", 0)
        return dict(judge(cdir, rc), dir=self.rel(cdir))

    @staticmethod
    def checked(s, a, chk):
        """The "episode" of an accepted check that is not recorded (record=False)."""
        return dict(name=f"seed_{s}_a{a}", dir=chk["dir"], seed=s, attempt_id=a, accepted=True, reasons=[],
                    state_log_sha256=chk["state_log_sha256"], check=chk["dir"], label="check")

    def record(self, s, a, plan_dir, chk):
        name = f"seed_{s}_a{a}"
        d = self.out / "episodes" / name
        rc = self.jobs.replay(self.task, plan_dir, a, d, self.out / "jobs" / f"record_{name}", self.fps)
        v, bc = judge(d, rc), read(d / "bag_check.json") or {}
        reasons = ([] if v["accepted"] else [f"replay exit {rc}: {v['failed_phase']} {v['reasons']}"]) + \
            ([] if v["state_log_sha256"] == chk["state_log_sha256"] else ["state_log.npz differs from the check's"]) + \
            ([] if bc.get("pass") is True else [f"bag check failed: {bc.get('failed')}"])
        for cam in self.cameras:               # ~0.6 GB of PNGs per episode; the bag holds them as JPEG
            shutil.rmtree(d / cam, ignore_errors=True)
        e = dict(name=name, dir=self.rel(d), seed=s, attempt_id=a, accepted=not reasons, reasons=reasons,
                 state_log_sha256=v["state_log_sha256"], check=chk["dir"], label="demo")
        if d.is_dir():
            h = read(plan_dir / "handoff.json") or {}
            row = next((r for r in h.get("attempts_detail", []) if r["attempt_id"] == a), None)
            J.write_json(d / "plan.json", dict(task=self.task, seed=s, plan_run=self.rel(plan_dir), attempt_id=a,
                                               attempt=row, curobo_seeds=h.get("curobo_seeds")))
            if (plan_dir / "handoff.npz").is_file():
                z = np.load(plan_dir / "handoff.npz")
                k = int(np.flatnonzero(z["attempt_id"] == a)[0])
                np.savez_compressed(d / "plan.npz", joint_names=z["joint_names"],
                                    **{x: z[x][k] for x in ("traj_robot_world", "door_angle", "tcp_world")})
            ep = read(d / "episode.json")
            if ep is not None:
                ep["datagen"] = {k: e[k] for k in ("seed", "attempt_id", "accepted", "label", "check")}
                st = h.get("starts") or {}
                if "sample" in st:          # a sampled start (reach starts block): the seed's sample + this attempt's
                    ep["datagen"]["start_sample"] = dict(st["sample"], mode=st.get("mode"),
                                                         attempt=next((x for x in st.get("attempts", [])
                                                                       if x["attempt_id"] == a), None))
                if s in self.layouts:
                    ep["datagen"]["layout"] = self.layouts[s]
                J.write_json(d / "episode.json", ep)
        return e


def set_variant(variant, task):
    """$REVOLVE_DATAGEN_VARIANT = variant (the launchers pass it on to every job), checked against generator.yaml."""
    import os
    from revolve_common import specs
    if variant:
        os.environ[specs.VARIANT_ENV] = variant
        specs_bridge.round_blocks(task)          # raises for a variant the task does not define for the robot


def generate(task, seeds, out, episodes_per_seed=1, workers=2, attempts=None, jobs=DOCKER, say=None, layouts=None,
             checks=1, record=True):
    kw = {} if say is None else dict(say=say)
    return Generation(task, seeds, out, episodes_per_seed, workers, attempts, jobs, layouts=layouts, checks=checks,
                      record=record, **kw).run()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", required=True, help="e.g. 0-9 or 0,3,5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes-per-seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2, help="seeds at a time (= planners on the GPU at once)")
    ap.add_argument("--attempts", type=int, default=None, help="trajopt problems per plan run (generator.yaml)")
    ap.add_argument("--variant", default=None, help="generator.yaml tasks.<task>.variants.<name> (an experiment's "
                                                     "datagen.variant): sets $REVOLVE_DATAGEN_VARIANT for the jobs")
    ap.add_argument("--checks", type=int, default=1, help="camera-off replays an attempt must pass, all of them (>= 1)")
    a = ap.parse_args(argv)
    if a.checks < 1:
        ap.error("--checks must be >= 1")
    set_variant(a.variant, a.task)
    m = generate(a.task, parse_seeds(a.seeds), a.out, a.episodes_per_seed, a.workers, a.attempts, checks=a.checks)
    return 1 if any(r["state"] == "error" for r in m["seeds"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
