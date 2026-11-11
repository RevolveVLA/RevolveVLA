"""Evaluation (host side, or the manager's container): closed-loop policy rollouts in the simulator, per seed and
per phase. Ported from the earlier pipeline's evaluate.py (run/report) with the legacy breakage fixed: no evals/*.yaml
(inputs are arguments: the experiment's eval block, a checkpoint, a seed list), the verdict is the task.yaml
phases judged live by the simulator (no closed_loop_success block), no subprocess PYTHONPATH reset (rollouts
are threads of this process), the wall-clock limit is derived from the sim budget and the measured real-time
factor (rollout.wall_limit) instead of 285 s.

  python components/evaluation/evaluate.py --task T --ckpt runs/<exp>/turn_<k>/ckpt --seeds 100-103 \
      --set heldout --out runs/<exp>/turn_<k>/eval --starts runs/<exp>/starts [--rollouts-per-seed 1] \
      [--retries 0] [--parallel N] [--policy smolvla|oracle] [--source DATAGEN_OUT ...] [--gui] [--video]
  evaluate(task, ckpt, seeds, out, set_name, starts_dir, ...) -> eval_summary dict

Per seed: its start (starts.py: the first replay-accepted attempt of the seed's plan run, cached in starts_dir),
then rollouts_per_seed rollouts (rollout.py: serve + Rosetta runner + the sim on a per-rollout internal network).
A rollout that fails phase P is retried up to `retries` times (experiment eval.retries, default 0): the policy
runs again from the failed attempt's phase_states/P_start.npz (restored in step 0). A stack error (no verdict)
is re-run once. `parallel` rollouts at a time (default: what the GPU memory holds, default_parallel).
--gui: watch the rollouts in the gz GUI on the host's X display (DISPLAY + XAUTHORITY); forces parallel 1.
--video: each rollout also records the scene camera into <rollout dir>/video.mp4 (not part of the verdict).

Layout under out/<set>/:
  <seed>_r<i>/            the first try (the recovery input: phase_report.json, phase_states/, episode.json, ...)
  <seed>_r<i>/retry_<j>/  retry j, restored from the previous try's failed phase start state
  eval_summary.json       seed x phase table (first try; final after retries), first-try success, all_pass
                          (every seed passed every phase on the first try), rollouts with each try, RTF, wall, GPU
A re-run resumes: a rollout dir whose rollout.json has a verdict is kept. PROGRESS {json} lines (rollouts,
rollouts_done, passed, first_try_passed, running) feed the wrapper's /status (--parser eval_progress:parse).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE)]
import rollout as R  # noqa: E402
import starts as S  # noqa: E402

from revolve_common import job as J  # noqa: E402

RESTORABLE = ("approach", "grasp", "open", "release")
RTF_FILE = REPO / "runs" / "eval_rtf.json"     # the last measured rollout RTF on this machine (per parallel count)
RTF_PRIOR = 0.1                                # legacy laptop RTF 0.12-0.15 with the policy server on the GPU
# Measured: one SmolVLA rollout (serve + sim with
# the 3 training cameras) peaks at about this much GPU memory; the oracle needs no server.
GPU_MIB_PER_ROLLOUT = {"smolvla": 1700, "oracle": 600}
GPU_RESERVE_MIB = 800


def gpu_mib():
    """(total, used) MiB of GPU 0 (nvidia-smi), or None."""
    p = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    if p.returncode or not p.stdout.strip():
        return None
    total, used = (int(x) for x in p.stdout.splitlines()[0].split(","))
    return total, used


def default_parallel(policy="smolvla", cap=4):
    """Rollouts that fit in the free GPU memory at the measured per-rollout peak (at least 1, at most cap)."""
    g = gpu_mib()
    if g is None:
        return 1
    free = g[0] - g[1] - GPU_RESERVE_MIB
    return max(1, min(cap, free // GPU_MIB_PER_ROLLOUT[policy]))


def measured_rtf(parallel):
    d = S.G.read(RTF_FILE) or {}
    return d.get(str(parallel)) or d.get("1") or RTF_PRIOR


class GpuSampler(threading.Thread):
    """GPU memory used [MiB] every second while the evaluation runs (all rollouts together)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.wait(1.0):
            g = gpu_mib()
            if g:
                self.samples.append(g[1])

    def summary(self):
        s = self.samples
        return dict(max_mib=max(s), median_mib=int(statistics.median(s)), samples=len(s)) if s else None


def done(rec):
    return rec is not None and rec.get("phases") is not None


class Evaluation:
    def __init__(self, task, ckpt, seeds, out, set_name, starts_dir, policy="smolvla", rollouts_per_seed=1, retries=0,
                 parallel=None, sources=(), budget_s=None, wait_s=None, after_success_s=None, rtf=None,
                 runner=None, gui=False, video=False, say=lambda m: print(m, flush=True)):
        self.task, self.ckpt, self.seeds, self.policy = task, ckpt, list(seeds), policy
        self.out = Path(out).resolve() / set_name
        self.set, self.starts_dir, self.sources = set_name, Path(starts_dir), list(sources)
        self.rps, self.retries = rollouts_per_seed, retries
        self.parallel = 1 if gui else parallel or default_parallel(policy)    # one GUI window at a time
        self.gui, self.video = gui, video
        self.budget = dict(budget_s=budget_s, wait_s=wait_s, after_success_s=after_success_s)
        self.rtf = rtf or measured_rtf(self.parallel)
        self.rtfs, self.lock = [], threading.Lock()
        self.runner = runner or R.run          # tests pass a stub
        self._say = say
        self.items = [(s, i) for s in self.seeds for i in range(self.rps)]
        self.state = {f"{s}_r{i}": "pending" for s, i in self.items}
        self.first = {}                        # rollout -> its first try's pass / fail

    def say(self, msg):
        self._say(f"{time.strftime('%H:%M:%S')} evaluate {self.set}: {msg}")

    def progress(self):
        st = list(self.state.values())
        print("PROGRESS " + json.dumps(dict(
            set=self.set, rollouts=len(st), rollouts_done=sum(x in ("pass", "fail", "error") for x in st),
            passed=sum(x == "pass" for x in st), first_try_passed=sum(x == "pass" for x in self.first.values()),
            running=[k for k, x in self.state.items() if x == "running"])), flush=True)

    def run(self):
        t0 = time.time()
        self.out.mkdir(parents=True, exist_ok=True)
        self.starts = S.resolve(self.task, self.seeds, self.starts_dir, self.sources, say=self.say)
        self.say(f"{len(self.items)} rollouts of {self.policy} on seeds {self.seeds}, {self.parallel} at a time, "
                 f"retries {self.retries}, assumed RTF {self.rtf:.3f}")
        self.progress()
        sampler = GpuSampler()
        sampler.start()
        try:
            with ThreadPoolExecutor(max_workers=self.parallel) as ex:
                results = list(ex.map(self.item, self.items))
        finally:
            sampler.stop.set()
        summary = self.summarize(results, sampler.summary(), round(time.time() - t0, 1))
        if self.rtfs:
            d = S.G.read(RTF_FILE) or {}
            d[str(self.parallel)] = round(min(self.rtfs), 3)
            J.write_json(RTF_FILE, d)
        return summary

    def one(self, start, d, seed, restore=None):
        """One rollout (re-run once on a stack error); a finished one is kept (resume)."""
        rec = S.G.read(d / "rollout.json")
        if done(rec) and (restore is None or rec.get("restore")):
            return rec
        for attempt in range(2):
            with self.lock:
                rtf = min(self.rtfs) if self.rtfs else self.rtf
            rec = self.runner(start, d, seed=seed, policy=self.policy, ckpt=self.ckpt, restore=restore, rtf=rtf,
                              task=self.task, say=self.say, gui=self.gui, video=self.video, **self.budget)
            if rec.get("rtf"):
                with self.lock:
                    self.rtfs.append(rec["rtf"])
            if done(rec):
                return rec
            self.say(f"{d.name}: stack error ({rec.get('error')}), attempt {attempt + 1} of 2")
        return rec

    def item(self, si):
        seed, i = si
        name = f"{seed}_r{i}"
        d = self.out / name
        start = REPO / self.starts[seed]["dir"]
        self.state[name] = "running"
        self.progress()
        tries = [self.one(start, d, seed)]
        self.first[name] = "pass" if tries[0].get("passed") else "fail"
        while not tries[-1].get("passed") and done(tries[-1]) and len(tries) <= self.retries:
            prev = tries[-1]
            ph = prev["failed_phase"]
            src = REPO / prev["dir"]
            if ph not in RESTORABLE or not (src / "phase_states" / f"{ph}_start.npz").is_file():
                break                       # a global failure or no saved state: nothing to restore
            tries.append(self.one(start, d / f"retry_{len(tries)}", seed, restore=(src, ph)))
        last = tries[-1]
        self.state[name] = "pass" if last.get("passed") else ("fail" if done(last) else "error")
        self.say(f"{name}: " + ", ".join(
            f"try {k}: " + ("PASS" if t.get("passed") else f"FAIL {t.get('failed_phase')}" if done(t) else
                            f"ERROR {t.get('error')}") for k, t in enumerate(tries)))
        self.progress()
        return dict(seed=seed, rollout=i, name=name, tries=tries)

    def summarize(self, results, gpu, wall_s):
        table, rows = {}, []
        for x in results:
            f, last = x["tries"][0], x["tries"][-1]
            row = dict(dir=f["dir"], first_try=f.get("phases"), passed_first_try=bool(f.get("passed")),
                       failed_phase=f.get("failed_phase"), error=f.get("error"), passed=bool(last.get("passed")),
                       tries=len(x["tries"]), final_failed_phase=last.get("failed_phase") if not last.get("passed") else None,
                       retries=[dict(dir=t["dir"], restored_at=(t.get("restore") or {}).get("phase"), phases=t.get("phases"),
                                     passed=bool(t.get("passed")), restore_check=t.get("restore_check"))
                                for t in x["tries"][1:]],
                       sim_s=f.get("sim_s"), rtf=f.get("rtf"), wall_s=f["timing"].get("rollout_wall_s"),
                       end_reason=f.get("end_reason"),
                       inference_ms_median=((f.get("server") or {}).get("inference_ms") or {}).get("median"))
            table.setdefault(str(x["seed"]), {})[f"r{x['rollout']}"] = row
            rows.append(row)
        n = len(rows)
        k1, k = sum(r["passed_first_try"] for r in rows), sum(r["passed"] for r in rows)
        num = lambda key: [r[key] for r in rows if r.get(key) is not None]
        s = dict(task=self.task, set=self.set, policy=self.policy, ckpt=str(self.ckpt) if self.ckpt else None,
                 seeds=self.seeds, rollouts_per_seed=self.rps, retries=self.retries, parallel=self.parallel,
                 starts={str(sd): v["dir"] for sd, v in self.starts.items()}, rollouts=n,
                 first_try_passed=k1, first_try_success=k1 / n if n else None, passed_with_retries=k,
                 errors=sum(r["error"] is not None and r["first_try"] is None for r in rows),
                 all_pass=n > 0 and k1 == n, all_pass_with_retries=n > 0 and k == n, table=table,
                 rtf=dict(assumed=self.rtf, measured_min=min(num("rtf")) if num("rtf") else None,
                          measured_median=statistics.median(num("rtf")) if num("rtf") else None),
                 rollout_wall_s=dict(median=statistics.median(num("wall_s")) if num("wall_s") else None,
                                     max=max(num("wall_s")) if num("wall_s") else None),
                 eval_wall_s=wall_s, gpu_memory=gpu, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        J.write_json(self.out / "eval_summary.json", s)
        self.say(f"done: first try {k1}/{n}, with retries {k}/{n}, all_pass {s['all_pass']}")
        return s


def evaluate(task, ckpt, seeds, out, set_name, starts_dir, **kw):
    return Evaluation(task, ckpt, seeds, out, set_name, starts_dir, **kw).run()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", default=None, help="the checkpoint dir (train's out: config.json + model.safetensors)")
    ap.add_argument("--seeds", required=True, help="100-103 / 1,5")
    ap.add_argument("--set", default="heldout", help="probe | heldout (the dir name under --out)")
    ap.add_argument("--out", required=True, help="runs/<exp>/turn_<k>/eval")
    ap.add_argument("--starts", required=True, help="runs/<exp>/starts (seed -> start, cached)")
    ap.add_argument("--source", nargs="*", default=[], help="datagen output dirs searched for starts first")
    ap.add_argument("--policy", default="smolvla", choices=R.POLICIES)
    ap.add_argument("--rollouts-per-seed", type=int, default=1)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--budget-s", type=float, default=None, help="override task.yaml rollout.budget_s (tests)")
    ap.add_argument("--wait-s", type=float, default=None)
    ap.add_argument("--gui", action="store_true", help="show the gz GUI on $DISPLAY (one rollout at a time)")
    ap.add_argument("--video", action="store_true", help="record the scene camera into <rollout dir>/video.mp4")
    a = ap.parse_args(argv)
    if a.policy == "smolvla" and not a.ckpt:
        ap.error("--ckpt is required for --policy smolvla")
    s = evaluate(a.task, Path(a.ckpt).resolve() if a.ckpt else None, S.G.parse_seeds(a.seeds), a.out, a.set, a.starts,
                 policy=a.policy, rollouts_per_seed=a.rollouts_per_seed, retries=a.retries, parallel=a.parallel,
                 sources=a.source, budget_s=a.budget_s, wait_s=a.wait_s, gui=a.gui, video=a.video)
    return 1 if s["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
