"""Capacity bench (GPU + docker; skipped unless RECOVERY_BENCH=1): how many slots (= warm sims) one host runs before the sim's real-time
factor or the verdicts degrade. For each N in RECOVERY_BENCH_N (default 1,2,3,4,6) one coupled evaluation on N slots: `noisy_expert`
rollouts with the DAgger recorder (takeover by the moveit expert) and automoma phase recovery of the failures, warm resets, on the
slots' own sims. Measured: RTF per rollout (sim seconds of the episode / wall seconds of its client container), wall per episode,
GPU memory per slot, GPU util, CPU, RAM, load, verdicts against N=1, infrastructure faults. N stops growing at the first RTF < 0.5, fault
or error. Numbers: RECOVERY_BENCH_NUMBERS=<file> (JSON). RECOVERY_BENCH_RUN=<dir> reuses a training run, RECOVERY_BENCH_KEEP=1 keeps the
outputs. Needs the GPU gate: >= 6 GB free and the other processes <= 20 % util."""
import json
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pytest
from bench_metrics import ClientTimes, Sampler, parse_gpu, rollout_rtfs, stability, summarise
from test_recovery_gpu import PREFIXES, REPO, TASK, Run, all_tries, leftovers, manifest_of, recovery_numbers, rel, resets_of

from datagen import cli as datagen_cli
from evaluation import cli as eval_cli
from revolve_common.sim_session import ClientContainer

pytestmark = [pytest.mark.bench,
              pytest.mark.skipif(os.environ.get("RECOVERY_BENCH") != "1", reason="needs a GPU: RECOVERY_BENCH=1")]

SLOTS = [int(n) for n in os.environ.get("RECOVERY_BENCH_N", "1,2,3,4,6").split(",")]
MIN_RTF, FREE_GPU_MIB, OTHERS_UTIL = 0.5, 6144, 20
CONFIGS, ROLLOUTS_PER_SLOT = 3, 2
NUMBERS: dict = {}
BASE: dict[str, set] = {}                      # verdicts of N=1 per config
STOPPED: list[str] = []                        # why the bench stopped growing


def gpu(query: str) -> str:
    return subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout


def gate() -> dict:
    """The host gate: free GPU memory and the util of the others (3 samples, 1 s apart; nothing of ours runs)."""
    utils = []
    for _ in range(3):
        utils.append(parse_gpu(gpu("utilization.gpu,memory.used"))[0])
        time.sleep(1)
    util, used = parse_gpu(gpu("utilization.gpu,memory.used"))
    total = float(gpu("memory.total").strip())
    return {"others_util_max": max(utils), "gpu_used_mib": used, "gpu_free_mib": total - used, "load1": os.getloadavg()[0]}


@pytest.fixture(scope="module")
def trained():
    assert not leftovers(), f"containers of another run: {leftovers()}"
    d = Path(os.environ.get("RECOVERY_BENCH_RUN", REPO / "runs" / f"recovery_bench_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"))
    d.mkdir(parents=True, exist_ok=True)
    run = Run(d)
    if not (d / "train" / "manifest.json").is_file():
        assert datagen_cli.main([run.train(CONFIGS), "all", "--workers", "2", "--headless"], root=REPO) == 0
    manifest = json.loads((d / "train" / "manifest.json").read_text())
    assert manifest["counts"]["passed"] >= CONFIGS and all(e["bag"] for e in manifest["episodes"] if e["passed"]), manifest["counts"]
    yield run
    if path := os.environ.get("RECOVERY_BENCH_NUMBERS"):
        Path(path).write_text(json.dumps(NUMBERS, indent=1, default=str))
    print("\nNUMBERS " + json.dumps(NUMBERS, default=str))
    if not os.environ.get("RECOVERY_BENCH_KEEP"):
        shutil.rmtree(d, ignore_errors=True)
    assert not leftovers(), f"containers left: {leftovers()}"


def sim_s_of(log: Path) -> float | None:
    try:
        return json.loads((log.parent / "episode.json").read_text())["result"]["duration_s"]
    except (OSError, ValueError, KeyError):
        return None


def verdicts(out: Path) -> dict[str, set]:
    found: dict[str, set] = defaultdict(set)
    for _, key, _, rec, _ in all_tries(out):
        found[key.rsplit("_r", 1)[0]].add((bool(rec["passed"]), rec.get("failed_phase")))
    return dict(found)


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.floor(q * len(xs)))]


@pytest.mark.parametrize("n", SLOTS)
def test_capacity_at_n_slots(trained, n):
    if STOPPED:
        pytest.skip(f"the bench stopped growing: {STOPPED[0]}")
    run, name, quota = trained, f"n{n}", 2 * n
    host = gate()
    assert host["gpu_free_mib"] >= FREE_GPU_MIB and host["others_util_max"] <= OTHERS_UTIL, f"GPU gate closed: {host}"
    recovery = run.write(f"{name}_recovery", {
        **TASK, "from": {"eval": rel(run.dir / name)},
        "triggers": [{"plugin": "verdict"}, {"plugin": "stall", "phases": ["approach"], "window_s": 3, "min_motion": 0.01}],
        "phases": {"restart_from": "before_failure", "back_s": 1.0, "max_object_joint": 0.02,
                   "generator": {"plugin": "automoma", "attempts": 3, "record": "images"}, "per_failure": 2,
                   "quota": {"approach": quota, "grasp": quota}},
        "live": {"plugin": "dagger", "triggers": [{"plugin": "phase_time", "fraction": 0.2}], "expert": {"plugin": "moveit_takeover"},
                 "record": "full", "quota": {"approach": quota, "grasp": quota, "open": quota}},
        "keep": "successes", "seed": 0, "out": rel(run.dir / name / "recovery")})
    rollouts = math.ceil(ROLLOUTS_PER_SLOT * n / CONFIGS)
    spec = run.evaluation(name, {"plugin": "noisy_expert", "sigma": 0.03}, configs=CONFIGS, rollouts=rollouts, workers=n,
                          record="images", run={"reset": "warm"}, recovery={"spec": recovery, "enabled": True})
    sampler, times = Sampler(), ClientTimes()
    sampler.start()
    t0 = time.monotonic()
    with times.patched(ClientContainer):
        code = eval_cli.main([spec, "all", "--headless"], root=REPO)
    wall = round(time.monotonic() - t0)
    samples = sampler.stop()
    assert code in (0, 1), code                                          # 1: a recovery quota left short
    out = run.dir / name
    rows = list(all_tries(out))
    errors = [r["error"] for *_, r, _ in rows if r.get("error")]
    rtfs = rollout_rtfs(times.runs, sim_s_of)
    jobs = [r for r in times.runs if Path(r["log"]).name != "client.log"]
    numbers = recovery_numbers(out / "recovery")
    manifest = manifest_of(out / "recovery")
    seen = verdicts(out)
    if n == SLOTS[0]:
        BASE.update(seen)
    got = {"slots": n, "host_before": host, "rollouts": len(rows), "wall_s": wall, "rollout_errors": errors,
           "client_faults": times.faults, "rtf": {"n": len(rtfs), "mean": round(mean(r["rtf"] for r in rtfs), 3) if rtfs else None,
                                                   "min": min((r["rtf"] for r in rtfs), default=None),
                                                   "p10": percentile([r["rtf"] for r in rtfs], 0.1) if rtfs else None},
           "wall_per_episode_s": round(mean(r["wall_s"] for r in rtfs), 1) if rtfs else None,
           "sim_s_per_episode": round(mean(r["sim_s"] for r in rtfs), 1) if rtfs else None,
           "phase_jobs": {"runs": len(jobs), "wall_s_mean": round(mean(r["wall_s"] for r in jobs), 1) if jobs else None},
           "recovery": numbers, "manifest_entries": len(manifest["episodes"]), "eval_resets": dict(resets_of(out)),
           "verdicts_vs_first": stability(BASE, seen), "sampler": summarise(samples, PREFIXES, host["gpu_used_mib"])}
    NUMBERS[name] = got
    print(f"BENCH {name} " + json.dumps(got, default=str))
    reasons = [f"N={n}: RTF mean {got['rtf']['mean']} < {MIN_RTF}" if got["rtf"]["mean"] is not None and got["rtf"]["mean"] < MIN_RTF else "",
               f"N={n}: {len(times.faults)} client fault(s)" if times.faults else "",
               f"N={n}: {len(errors)} rollout error(s)" if errors else "",
               f"N={n}: a phase job failed" if "error" in numbers["jobs"] else ""]
    if reason := next((r for r in reasons if r), None):
        STOPPED.append(reason)
