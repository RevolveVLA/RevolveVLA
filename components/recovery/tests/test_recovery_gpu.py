"""The recovery component on the real simulator (GPU + docker; skipped unless RECOVERY_GPU=1; needs the simulator image, the assets
and a free GPU). open_fridge_door / mobile_panda. About 15 min with two sim slots; outputs in runs/recovery_gpu_<ts>/ (deleted
afterwards, RECOVERY_GPU_KEEP=1 keeps them; RECOVERY_GPU_RUN=<such a dir> reuses its training run; RECOVERY_GPU_NUMBERS=<file> writes
the numbers as JSON).

A small training run (the moveit expert) is the failing policies' replay source, then
 - coupled: `noisy_expert` rolled out by the evaluation on 2 warm slots with the DAgger recorder and automoma phase recovery on the
   same slots (the evaluation's `recovery:` key); its recovery manifest and bags follow the datagen contract;
 - oracle: the bag replay, evaluated plain, then phase recovery with the moveit generator and warm reset on its failures
   (unreproducible ones counted, quota respected);
 - triggers: `frozen` (never moves) once per trigger type (phase_time, stall, condition): each fires, the expert takes over,
   the sidecar is valid, the takeover is smooth (max_delta_action);
 - the recipe resolves over those outputs.
Further tests: Ctrl-C / SIGTERM of a phase recovery and of a coupled evaluation (nothing left, state resumable), two ./revolve
processes started together (distinct ROS domains), and the SO-101 XL drawer (phase recovery, DAgger, coupled; moveit generator).
The moveit expert itself passes only about half of its episodes (its `open` step), so "recovered" is reported twice: the expert got
the episode past the phase it took over (the recovery's job) and the whole episode passed. Verdicts are never compared one by one."""
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest
import yaml

from datagen import cli as datagen_cli
from datagen import contract
from evaluation import cli as eval_cli
from recipe import cli as recipe_cli
from recovery import cli as recovery_cli
from recovery.core import flags
from recovery.core.live import tries
from revolve_common.sim_session import LEASE_DIR, ClientContainer, Docker

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("RECOVERY_GPU") != "1", reason="needs a GPU: RECOVERY_GPU=1")]

REPO = Path(__file__).resolve().parents[3]
PREFIXES = ("dg-", "ev-", "rc-", "acc-")
SMOOTH_RAD = 0.1                         # the expert's largest 30 Hz joint step in its first second after a takeover
TASK = {"task": "open_fridge_door", "robot": "mobile_panda"}
NUMBERS: dict = {}


def leftovers():
    return [n for n in Docker().names() if n.startswith(PREFIXES)]


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


class Run:
    """The run dir with spec writers; every run of this module goes through `cli`/`recover`/`evaluate` as `./revolve` does."""

    def __init__(self, d: Path):
        self.dir = d

    def write(self, name: str, data: dict) -> str:
        path = self.dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return rel(path)

    def train(self, scenes: int) -> str:
        return self.write("train", {**TASK, "out": rel(self.dir / "train"), "seed": 0,
                                    "generator": {"plugin": "moveit", "attempts": 3, "episodes_per_scene": 1, "record": "no-images"},
                                    "randomisers": {"robot_start": {"x_m": [-1.85, -1.65], "y_m": [1.0, 1.2], "yaw_rad": [1.47, 1.67],
                                                                    "arm": {"choices": ["folded"]}}},
                                    "validators": [{"plugin": "collision_free"}],
                                    "splits": {"train": {"target": scenes, "pool": {"initial": [0, scenes + 2],
                                                                                    "replacement": [scenes + 3, scenes + 8]}}}})

    def evaluation(self, name: str, policy: dict, configs: int, rollouts: int, workers: int, **extra) -> str:
        return self.write(name, {"train": rel(self.dir / "train.yaml"), "policy": policy, "out": rel(self.dir / name),
                                 "sets": {"train": {"from": "train", "count": configs}}, "retries": 0, "rollouts": rollouts,
                                 "rollout": {"plugin": "sim", "record": extra.pop("record", "no-images")},
                                 "run": {"workers": workers, "headless": True, **extra.pop("run", {})}, **extra})


@pytest.fixture(scope="module")
def run():
    assert not leftovers(), f"containers of another run: {leftovers()}"
    d = Path(os.environ.get("RECOVERY_GPU_RUN", REPO / "runs" / f"recovery_gpu_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"))
    d.mkdir(parents=True, exist_ok=True)
    yield Run(d)
    if numbers := os.environ.get("RECOVERY_GPU_NUMBERS"):
        Path(numbers).write_text(json.dumps(NUMBERS, indent=1, default=str))
    print("\nNUMBERS " + json.dumps(NUMBERS, default=str))
    if not os.environ.get("RECOVERY_GPU_KEEP"):
        shutil.rmtree(d, ignore_errors=True)
    assert not leftovers(), f"containers left: {leftovers()}"


@pytest.fixture(scope="module")
def trained(run):
    """The training run whose episodes the oracle and the bad policies replay."""
    t0 = time.monotonic()
    spec = run.train(3)
    if not (run.dir / "train" / "manifest.json").is_file():
        assert datagen_cli.main([spec, "all", "--workers", "2", "--headless"], root=REPO) == 0
    manifest = json.loads((run.dir / "train" / "manifest.json").read_text())
    NUMBERS["train"] = {"accepted": manifest["counts"]["passed"], "attempted": manifest["counts"]["attempted"],
                        "wall_s": round(time.monotonic() - t0)}
    assert manifest["counts"]["passed"] >= 3
    unrecorded = [(e["seed"], e["episode"]) for e in manifest["episodes"] if e["passed"] and not e["bag"]]
    assert not unrecorded, f"the simulator's recorder dropped the bag of accepted episodes {unrecorded} (a host under load): the replays need them"
    return run


def manifest_of(out: Path) -> dict:
    assert contract.check_manifest(out) == [], contract.check_manifest(out)
    return json.loads((out / "manifest.json").read_text())


def resets_of(eval_out: Path) -> Counter:
    """The warm resets of an evaluation run's rollouts: hit (reached), miss (fell back to a sim of its own), none (first on a sim)."""
    found: Counter = Counter()
    for _, _, _, rec, _ in all_tries(eval_out):
        r = rec.get("reset") or {}
        found["hit" if r.get("mode") == "warm" and (r.get("report") or {}).get("ok") else
              "miss" if r.get("mode") == "warm" and r.get("report") else "none"] += 1
    return found


def all_tries(eval_out: Path):
    """(set, rollout key, try, record, sidecar | None) of every try of an evaluation run (`tries` lists only those with a sidecar)."""
    results = json.loads((eval_out / "results.json").read_text())
    for set_name, block in sorted(results["sets"].items()):
        for key, entry in sorted(block["rollouts"].items()):
            for i, t in enumerate(entry["tries"]):
                yield set_name, key, i, t["record"], flags.read(t["record"]["result_file"]) if t["record"].get("result_file") else None


def phase_resets(out: Path) -> Counter:
    """The warm resets of recovery jobs (resets.json per job): hit / miss."""
    found: Counter = Counter()
    for path in out.glob("*/*/resets.json"):
        for r in json.loads(path.read_text()):
            found["hit" if r["ok"] else "miss"] += 1
    return found


def recovery_numbers(out: Path) -> dict:
    state = json.loads((out / "state.json").read_text())
    manifest = manifest_of(out)
    rows = {}
    for phase in ("approach", "grasp", "open"):
        fails = [f for f in state.get("failures", []) if f["phase"] == phase]
        rows[phase] = {"failures": len(fails), "unreproducible": sum(f["restart"]["reason"] is not None for f in fails),
                       "recorded": sum(e["recovery"]["phase"] == phase and e["recovery"]["kind"] == "phase" for e in manifest["episodes"]),
                       "passed": sum(e["passed"] and e["recovery"]["phase"] == phase and e["recovery"]["kind"] == "phase"
                                     for e in manifest["episodes"])}
    return {"phases": rows, "jobs": dict(Counter(j["status"] for j in state.get("jobs", {}).values())),
            "resets": dict(phase_resets(out)), "scan": state.get("scan", {})}


def assert_gate(out: Path):
    """A failure is reproducible exactly when the bag had the robot state and no scene joint beyond the gate (the door closed)."""
    for f in json.loads((out / "state.json").read_text()).get("failures", []):
        r = f["restart"]
        assert (r["reason"] is None) == (r["base"] is not None and r["object_joint"] <= 0.02), f


def assert_quota_respected(out: Path, quota: dict, rows: dict):
    for phase, row in rows.items():
        assert row["recorded"] <= quota.get(phase, 0), (phase, row, quota)       # a phase not listed is not recovered
        assert row["unreproducible"] <= row["failures"]


def bag_contract(manifest_out: Path, record: str, mounts=()):
    """datagen.contract's bag check (frozen topics, 30 Hz action series, cameras) in a client container."""
    client = ClientContainer(f"acc-{os.getpid()}", 97, ["python3", "-m", "datagen.contract", "--out", str(manifest_out), "--record", record],
                             manifest_out, pythonpath=["/repo/components/datagen/src"], mounts=mounts)
    log = manifest_out / "contract.log"
    rc = client.run(log, timeout=900)
    lines = [ln for ln in log.read_text().splitlines() if ln.startswith("RESULT ")]
    result = json.loads(lines[-1][len("RESULT "):]) if lines else {"problems": [f"no RESULT line: see {log}"]}
    assert rc == 0 and result["problems"] == [], result["problems"]
    return len(result["bags"])


# ---------------------------------------------------------------- coupled: DAgger + phase recovery on the evaluation's slots
def test_the_coupled_run_records_takeovers_and_phase_episodes_on_two_warm_slots(trained):
    run = trained
    recovery = run.write("coupled_recovery", {
        **TASK, "from": {"eval": rel(run.dir / "coupled")},
        "triggers": [{"plugin": "verdict"}, {"plugin": "stall", "phases": ["approach"], "window_s": 3, "min_motion": 0.01}],
        "phases": {"restart_from": "before_failure", "back_s": 1.0, "max_object_joint": 0.02,
                   "generator": {"plugin": "automoma", "attempts": 3, "record": "images"}, "per_failure": 2,
                   "quota": {"approach": 4, "grasp": 4}},
        "live": {"plugin": "dagger", "triggers": [{"plugin": "phase_time", "fraction": 0.2}, {"plugin": "stall", "window_s": 2, "min_motion": 0.01}],
                 "expert": {"plugin": "moveit_takeover"}, "record": "full", "quota": {"approach": 3, "grasp": 3, "open": 3}},
        "keep": "successes", "seed": 0, "out": rel(run.dir / "coupled" / "recovery")})
    spec = run.evaluation("coupled", {"plugin": "noisy_expert", "sigma": 0.03}, configs=3, rollouts=2, workers=2,
                          record="images", run={"reset": "warm"}, recovery={"spec": recovery, "enabled": True})
    t0 = time.monotonic()
    assert eval_cli.main([spec, "all", "--headless"], root=REPO) in (0, 1)          # 1: a recovery quota left short (checked below)
    wall = round(time.monotonic() - t0)
    out = run.dir / "coupled"
    rows = list(all_tries(out))
    assert len(rows) == 6 and not [r for *_, r, _ in rows if r.get("error")], "rollouts with an error"
    live = [(r, f) for *_, r, f in rows if f and f["takeover_t"] is not None]
    assert live and all(flags.read(r["result_file"])["segments"][-1][2] == "expert" for r, _ in live)
    taken = Counter(f["phase"] for _, f in live)
    assert all(n <= {"approach": 3, "grasp": 3, "open": 3}[p] + 1 for p, n in taken.items()), taken        # quota + workers - 1
    rec_out = out / "recovery"
    numbers = recovery_numbers(rec_out)
    assert_quota_respected(rec_out, {"approach": 4, "grasp": 4}, numbers["phases"])
    assert_gate(rec_out)
    assert "error" not in numbers["jobs"], numbers
    manifest = manifest_of(rec_out)
    assert {e["recovery"]["kind"] for e in manifest["episodes"]} <= {"phase", "dagger"} and manifest["episodes"]
    bags = bag_contract(rec_out, "images", mounts=[run.dir / "train", out])
    NUMBERS["coupled"] = {"wall_s": wall, "rollouts": len(rows), "takeovers": dict(taken), "triggers": dict(Counter(f["trigger"] for _, f in live)),
                          "expert_reached_open_or_passed": sum(r["passed"] or json.loads(Path(r["result_file"]).read_text())["failed_phase"] == "open"
                                                               for r, _ in live),
                          "expert_passed": sum(r["passed"] for r, _ in live),
                          "max_delta_action": [round(f["max_delta_action"], 4) for _, f in live if f["max_delta_action"] is not None],
                          "eval_resets": dict(resets_of(out)), "phase": numbers, "manifest_entries": len(manifest["episodes"]),
                          "contract_bags": bags}


# ---------------------------------------------------------------- oracle: the bag replay, then phase recovery with moveit
def test_oracle_failures_are_recovered_by_phase_with_moveit_and_warm_reset(trained):
    run = trained
    t0 = time.monotonic()
    assert eval_cli.main([run.evaluation("oracle", {"plugin": "oracle"}, configs=3, rollouts=2, workers=2, run={"reset": "warm"}), "all",
                          "--headless"], root=REPO) == 0
    eval_wall = round(time.monotonic() - t0)
    oracle = run.dir / "oracle"
    rows = list(all_tries(oracle))
    assert len(rows) == 6 and not [r for *_, r, _ in rows if r.get("error")]
    results = {r["episode"]: json.loads(Path(r["result_file"]).read_text()) for *_, r, _ in rows}
    failed = Counter(x["failed_phase"] for x in results.values() if not x["passed"])
    hits = resets_of(oracle)
    assert hits["hit"] >= 1, f"no rollout was reached by a warm reset: {hits}"                  # 6 rollouts on 2 sims
    quota = {"approach": 2, "grasp": 2, "open": 1}
    spec = run.write("oracle_recovery", {
        **TASK, "from": {"eval": rel(oracle)}, "triggers": [{"plugin": "verdict"}],
        "phases": {"restart_from": "phase_start", "generator": {"plugin": "moveit", "attempts": 3, "record": "no-images"},
                   "per_failure": 1, "quota": quota},
        "keep": "successes", "seed": 0, "run": {"workers": 2, "reset": "warm", "headless": True}, "out": rel(run.dir / "oracle_recovery")})
    t0 = time.monotonic()
    code = recovery_cli.main([str(REPO / spec), "all", "--headless"], root=REPO)
    wall = round(time.monotonic() - t0)
    out = run.dir / "oracle_recovery"
    numbers = recovery_numbers(out)
    assert code in (0, 1) and "error" not in numbers["jobs"], (code, numbers)
    assert_quota_respected(out, quota, numbers["phases"])
    assert_gate(out)
    manifest = manifest_of(out)
    assert all(e["recovery"]["kind"] == "phase" and e["recovery"]["reset"]["mode"] in ("warm", "relaunch") for e in manifest["episodes"])
    NUMBERS["oracle"] = {"eval_wall_s": eval_wall, "failed_phases": dict(failed), "passed": sum(x["passed"] for x in results.values()),
                         "eval_resets": dict(hits), "recovery_wall_s": wall, "phase": numbers, "manifest_entries": len(manifest["episodes"])}
    NUMBERS["jobs_all"] = NUMBERS["coupled"]["phase"]["jobs"] if "coupled" in NUMBERS else {}
    assert sum(numbers["jobs"].values()) + sum(NUMBERS["jobs_all"].values()) >= 1, "no phase job ran in this module"


# ---------------------------------------------------------------- every trigger type fires; the expert takes over smoothly
TRIGGERS = {"phase_time": {"plugin": "phase_time", "fraction": 0.1},
            "stall": {"plugin": "stall", "window_s": 1.5, "min_motion": 0.01},
            "condition": {"plugin": "condition", "value": "approach.done[0]", "op": "gt", "threshold": 0.5, "for_s": 1.0}}


@pytest.fixture(scope="module")
def takeovers(trained):
    """One `frozen` live run per trigger type (one config each, one sim at a time) -> [(trigger name, sidecar, episode record, result)]."""
    run, found = trained, []
    for name, trigger in TRIGGERS.items():
        ev = run.evaluation(f"{name}_eval", {"plugin": "frozen"}, configs=1, rollouts=1, workers=1)
        spec = run.write(f"{name}_live", {**TASK, "from": {"eval": ev}, "live": {"plugin": "dagger", "triggers": [trigger], "record": "full"},
                                          "keep": "all", "seed": 0, "run": {"workers": 1, "reset": "relaunch", "headless": True},
                                          "out": rel(run.dir / f"{name}_live")})
        assert recovery_cli.main([str(REPO / spec), "live", "--headless"], root=REPO) == 0
        out = run.dir / f"{name}_live"
        (set_name, key, i, rec, f), = list(tries(out / "live"))
        found.append((name, f, rec, json.loads(Path(rec["result_file"]).read_text()), out))
    return found


def test_every_trigger_type_fires_and_the_expert_recovers_smoothly(takeovers):
    fired = {f["trigger"] for _, f, *_ in takeovers}
    assert fired == set(TRIGGERS), f"fired: {fired}"
    for name, f, rec, result, out in takeovers:
        assert f["trigger"] == name and f["takeover_t"] is not None and f["phase"] == "approach" and f["expert"], f
        assert f["segments"][0][2] == "policy" and f["segments"][-1][2] == "expert" and f["segments"][-1][1] == pytest.approx(result["duration_s"], abs=0.01)
        assert f["max_delta_action"] is not None and f["max_delta_action"] < SMOOTH_RAD, f
    got_past = sum(r["passed"] or x["failed_phase"] == "open" for _, _, r, x, _ in takeovers)
    assert got_past >= 2 * len(takeovers) / 3, f"the expert got only {got_past}/{len(takeovers)} episodes past the approach it took over"
    manifest = [manifest_of(out) for *_, out in takeovers]
    assert all(m["episodes"] and m["episodes"][0]["recovery"]["kind"] == "dagger" for m in manifest)
    NUMBERS["triggers"] = {name: {"takeover_t": f["takeover_t"], "reason": f["reason"], "max_delta_action": round(f["max_delta_action"], 4),
                                  "expert_passed": rec["passed"], "failed_phase": x["failed_phase"]} for name, f, rec, x, _ in takeovers}
    NUMBERS["triggers"]["_summary"] = {"fired": sorted(fired), "passed": sum(r["passed"] for _, _, r, _, _ in takeovers),
                                      "past_approach": got_past, "episodes": len(takeovers)}


# ---------------------------------------------------------------- the recipe over the outputs
def test_the_recipe_resolves_over_the_outputs(trained, takeovers):
    run = trained
    oracle, coupled = run.dir / "oracle", run.dir / "coupled"
    phase_runs = [r for r in (run.dir / "oracle_recovery", coupled / "recovery") if (r / "manifest.json").is_file()]
    kinds = {r: Counter(e["recovery"]["kind"] for e in json.loads((r / "manifest.json").read_text())["episodes"]) for r in phase_runs}
    phase = next((r for r in phase_runs if kinds[r]["phase"]), None)
    dagger = takeovers[0][-1]
    sources = {"base": {"plugin": "datagen", "manifest": rel(run.dir / "train" / "manifest.json")},
               "rollouts": {"plugin": "rollouts", "run": rel(oracle), "passed": True},
               "dagger": {"plugin": "recovery", "run": rel(dagger), "kind": "dagger", "segment": "interventions"}}
    take = {"base": "all", "rollouts": 1, "dagger": {"approach": 1}}
    if phase:
        sources["recovery"], take["recovery"] = {"plugin": "recovery", "run": rel(phase), "kind": "phase"}, {"approach": 1}
    spec = run.write("recipe", {"seed": 0, "unit": "episodes", "sources": sources, "take": take, "splits": {"train": 0.5, "val": 0.5},
                                "on_short": "warn", "out": rel(run.dir / "recipe")})
    assert recipe_cli.main([spec, "materialise"], root=REPO) == 0
    selection = json.loads((run.dir / "recipe" / "selection.json").read_text())
    got = {(c["source"], c["phase"]): c for c in selection["counts"]}
    assert got[("base", None)]["got"] == got[("base", None)]["available"] > 0
    for (source, phase_name), c in got.items():
        asked = take[source].get(phase_name) if isinstance(take[source], dict) else take[source]
        assert c["got"] == (c["available"] if asked == "all" else min(asked, c["available"])), c            # the take matches the ask
        assert sum(c["splits"].values()) == c["got"]
    assert not any(selection["refused"].values())
    NUMBERS["recipe"] = {f"{s}/{p}": {k: c[k] for k in ("asked", "got", "available", "splits")} for (s, p), c in got.items()}


# ---------------------------------------------------------------- real ./revolve processes: abort, resume, parallel runs
SIMS = re.compile(r"^(ev|rc)-[0-9a-f]{6}-\d+(-client)?$")          # the sims of an evaluation / recovery run (not the scan container)


def sims() -> list[str]:
    return [n for n in Docker().names() if SIMS.match(n)]


def start(run: Run, name: str, *args: str) -> subprocess.Popen:
    """`./revolve <args>` as its own process (log: <run dir>/<name>.log)."""
    with open(run.dir / f"{name}.log", "w") as log:
        return subprocess.Popen([sys.executable, str(REPO / "revolve"), *args], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)


def wait_for(proc: subprocess.Popen, condition, timeout: float, what: str):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return
        assert proc.poll() is None, f"the process ended with {proc.returncode} before {what}"
        time.sleep(1)
    proc.send_signal(signal.SIGINT)
    pytest.fail(f"no {what} in {timeout:.0f} s")


def held_leases() -> set[str]:
    """The ROS domains whose lease file is locked by some process now."""
    held = set()
    for path in LEASE_DIR.glob("*"):
        with open(path, "a+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                held.add(path.name)
    return held


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def stamps(root: Path, pattern: str) -> dict:
    return {str(p): p.stat().st_mtime_ns for p in sorted(root.glob(pattern)) if p.is_file()}


def committed_jobs(out: Path) -> dict:
    return {k: j for k, j in ((read_json(out / "state.json") or {}).get("jobs", {})).items() if j["status"] in ("done", "short")}


def abort(proc: subprocess.Popen, signum: int, leases_before: set):
    """Signal `proc`: it exits 128 + signum having removed its containers and released its leases."""
    proc.send_signal(signum)
    assert proc.wait(timeout=180) == 128 + signum
    assert not leftovers(), f"containers left: {leftovers()}"
    assert held_leases() <= leases_before, f"leases still held: {held_leases() - leases_before}"


def assert_state_consistent(out: Path):
    state = read_json(out / "state.json")
    assert state is not None, "state.json unreadable"
    assert set(state.get("jobs", {})) <= set(state.get("selection", [])) <= {f["id"] for f in state.get("failures", [])}, state
    assert all(j["status"] in ("done", "short", "error") for j in state.get("jobs", {}).values()), state["jobs"]


def assert_resumed(proc: subprocess.Popen, out: Path, before: dict, stamped: dict):
    """The re-run finishes every selected job and leaves the jobs committed before the abort as they were."""
    assert proc.wait(timeout=1200) in (0, 1)
    state = read_json(out / "state.json")
    assert set(state["selection"]) == set(state["jobs"]) and not leftovers(), state
    assert {k: state["jobs"][k] for k in before} == before, "a committed job changed"
    assert {k: v for k, v in stamps(out, "*/*/**/*").items() if k in stamped} == stamped, "a committed job's files were rewritten"
    manifest_of(out)


@pytest.fixture(scope="module")
def failing(trained):
    """An evaluation of `frozen` (it never moves: each rollout fails its approach) on 3 configs: what the phase recovery works on."""
    run = trained
    assert eval_cli.main([run.evaluation("failing", {"plugin": "frozen"}, configs=3, rollouts=1, workers=2), "all", "--headless"],
                         root=REPO) in (0, 1)
    failed = Counter(json.loads(Path(r["result_file"]).read_text())["failed_phase"] for *_, r, _ in all_tries(run.dir / "failing"))
    assert failed["approach"] == 3, failed
    return run


def phase_spec(run: Run, name: str, eval_out: Path, out: Path, **phases) -> str:
    return run.write(name, {**TASK, "from": {"eval": rel(eval_out)}, "triggers": [{"plugin": "verdict"}],
                            "phases": {"restart_from": "phase_start", "generator": {"plugin": "moveit", "attempts": 3, "record": "no-images"},
                                       "per_failure": 1, "quota": {"approach": 3}, **phases},
                            "keep": "successes", "seed": 0, "run": {"workers": 2, "reset": "relaunch", "headless": True}, "out": rel(out)})


def test_sigint_stops_a_phase_recovery_cleanly_and_a_rerun_resumes(failing):
    run, out = failing, failing.dir / "abort_recovery"
    spec, before = phase_spec(failing, "abort_recovery", failing.dir / "failing", out), held_leases()
    proc = start(run, "abort_recovery", "recover", spec, "all", "--headless")
    wait_for(proc, lambda: committed_jobs(out) and any(n.startswith("rc-") for n in sims()), 900, "a committed job and a running one")
    abort(proc, signal.SIGINT, before)
    assert_state_consistent(out)
    done = committed_jobs(out)
    assert done
    stamped = {k: v for j in done for k, v in stamps(out, f"*/{j}/**/*").items()}
    assert_resumed(start(run, "abort_recovery_rerun", "recover", spec, "all", "--headless"), out, done, stamped)
    NUMBERS["abort_phase"] = {"committed_at_abort": len(done), "jobs": dict(Counter(j["status"] for j in read_json(out / "state.json")["jobs"].values()))}


def test_sigterm_stops_a_coupled_evaluation_cleanly_and_a_rerun_resumes(trained):
    run, out = trained, trained.dir / "abort_coupled"
    recovery = phase_spec(run, "abort_coupled_recovery", out, out / "recovery")
    spec = run.evaluation("abort_coupled", {"plugin": "frozen"}, configs=3, rollouts=1, workers=2, recovery={"spec": recovery, "enabled": True})
    before = held_leases()
    proc = start(run, "abort_coupled", "evaluate", spec, "all", "--headless")
    wait_for(proc, lambda: any(n.startswith("rc-") for n in sims()), 900, "a recovery job on a slot")
    abort(proc, signal.SIGTERM, before)
    assert read_json(out / "results.json") is not None
    assert_state_consistent(out / "recovery")
    done, rolled = committed_jobs(out / "recovery"), {r["result_file"]: Path(r["result_file"]).stat().st_mtime_ns for *_, r, _ in all_tries(out)}
    stamped = {k: v for j in done for k, v in stamps(out / "recovery", f"*/{j}/**/*").items()}
    rerun = start(run, "abort_coupled_rerun", "evaluate", spec, "all", "--headless")
    if done:
        assert_resumed(rerun, out / "recovery", done, stamped)
    else:
        assert rerun.wait(timeout=1200) in (0, 1) and not leftovers()
    assert len(list(all_tries(out))) == 3
    assert {k: Path(k).stat().st_mtime_ns for k in rolled} == rolled, "a committed rollout was rerun"
    NUMBERS["abort_coupled"] = {"committed_jobs_at_abort": len(done), "rollouts_at_abort": len(rolled)}


def test_two_processes_started_together_lease_different_ros_domains(trained):
    run, procs = trained, {}
    for name in ("domain_a", "domain_b"):
        ev = run.evaluation(f"{name}_eval", {"plugin": "frozen"}, configs=1, rollouts=1, workers=1)
        procs[name] = run.write(name, {**TASK, "from": {"eval": ev}, "live": {"plugin": "dagger", "triggers": [TRIGGERS["phase_time"]], "record": "full"},
                                       "keep": "all", "seed": 0, "run": {"workers": 1, "reset": "relaunch", "headless": True},
                                       "out": rel(run.dir / f"{name}_live")})
    running = {name: start(run, name, "recover", spec, "live", "--headless") for name, spec in procs.items()}          # back to back
    domains = {}
    for name, proc in running.items():
        assert proc.wait(timeout=900) == 0, f"{name}: see {run.dir / name}.log"
        logs = (run.dir / f"{name}_live").rglob("*.log")
        domains[name] = {int(d) for log in logs for d in re.findall(r"--domain-id (\d+)", log.read_text(errors="replace"))}
        assert domains[name], f"no sim launch found in the logs of {name}"
        assert manifest_of(run.dir / f"{name}_live")["episodes"]
        bag_contract(run.dir / f"{name}_live", "no-images")
    assert domains["domain_a"].isdisjoint(domains["domain_b"]), domains
    assert not leftovers()
    NUMBERS["domains"] = {k: sorted(v) for k, v in domains.items()}


# ---------------------------------------------------------------- open_drawer / mobile_so101_xl (the moveit generator; AutoMoMa plans the fridge only)
DRAWER = {"task": "open_drawer", "robot": "mobile_so101_xl"}
MERGED = ("phases", "live", "run", "rollout")           # keys whose given fields replace only those fields of the spec's block


def derive(rn: Run, source: str, name: str, **changes) -> str:
    """specs/<source> with `changes` on top (None drops a key; a MERGED block is updated field by field) -> its path in the run dir."""
    data = yaml.safe_load((REPO / source).read_text())
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = {**data[key], **value} if key in MERGED and key in data else value
    return rn.write(name, data)


@pytest.fixture(scope="module")
def drawer(run):
    """A 2-scene training run of the drawer (its state and bags are what the eval sets and the noisy replay read) -> its spec."""
    base = yaml.safe_load((REPO / "specs/datagen/drawer_so101xl_v2.yaml").read_text())
    spec = derive(run, "specs/datagen/drawer_so101xl_v2.yaml", "drawer_train", out=rel(run.dir / "drawer_train"), generator={**base["generator"], "record": "no-images"},
                  splits={"train": {**base["splits"]["train"], "target": 2, "pool": {"initial": [0, 4], "replacement": [5, 9]}}})
    if not (run.dir / "drawer_train" / "manifest.json").is_file():
        assert datagen_cli.main([spec, "all", "--workers", "2", "--headless"], root=REPO) == 0
    manifest = json.loads((run.dir / "drawer_train" / "manifest.json").read_text())
    assert manifest["counts"]["passed"] >= 2 and not [e for e in manifest["episodes"] if e["passed"] and not e["bag"]], manifest["counts"]
    return spec


def drawer_eval(rn: Run, train: str, name: str, policy: dict, configs: int, **changes) -> str:
    return derive(rn, "specs/eval/drawer_so101xl_recovery.yaml", name, train=train, policy=policy, out=rel(rn.dir / name),
                  sets={"train": {"from": "train", "count": configs}}, rollouts=1, **changes)


def test_drawer_failures_are_recovered_by_phase_with_moveit(run, drawer):
    ev = drawer_eval(run, drawer, "drawer_frozen", {"plugin": "frozen"}, 2, recovery=None, run={"workers": 2, "reset": "relaunch"})
    assert eval_cli.main([ev, "all", "--headless"], root=REPO) in (0, 1)
    rows = list(all_tries(run.dir / "drawer_frozen"))
    assert len(rows) == 2 and not [r for *_, r, _ in rows if r.get("error")]
    out, quota = run.dir / "drawer_recovery", {"approach": 2, "grasp": 0, "open": 0}
    spec = derive(run, "specs/recovery/drawer_so101xl.yaml", "drawer_recovery", **{"from": {"eval": rel(run.dir / "drawer_frozen")}},
                  phases={"per_failure": 1, "quota": quota}, out=rel(out))
    t0 = time.monotonic()
    code = recovery_cli.main([str(REPO / spec), "all", "--headless"], root=REPO)
    numbers = recovery_numbers(out)
    assert code in (0, 1) and "error" not in numbers["jobs"], (code, numbers)
    assert_quota_respected(out, quota, numbers["phases"])
    assert_gate(out)
    manifest = manifest_of(out)
    assert numbers["phases"]["approach"]["recorded"] >= 1 and {e["recovery"]["kind"] for e in manifest["episodes"]} == {"phase"}
    NUMBERS["drawer_phase"] = {"wall_s": round(time.monotonic() - t0), "phase": numbers, "contract_bags": bag_contract(out, "no-images")}


def test_drawer_dagger_takes_over_a_frozen_policy_with_moveit(run, drawer):
    ev = drawer_eval(run, drawer, "drawer_dagger_eval", {"plugin": "frozen"}, 1, recovery=None, run={"workers": 1, "reset": "relaunch"})
    out = run.dir / "drawer_dagger"
    spec = derive(run, "specs/recovery/drawer_so101xl_dagger.yaml", "drawer_dagger", **{"from": {"eval": ev}},
                  live={"triggers": [TRIGGERS["phase_time"]]}, out=rel(out))
    assert recovery_cli.main([str(REPO / spec), "live", "--headless"], root=REPO) == 0
    (set_name, key, i, rec, f), = list(tries(out / "live"))
    result = json.loads(Path(rec["result_file"]).read_text())
    assert f["trigger"] == "phase_time" and f["takeover_t"] is not None and f["phase"] == "approach" and f["expert"], f
    assert f["segments"][0][2] == "policy" and f["segments"][-1][2] == "expert" and f["segments"][-1][1] == pytest.approx(result["duration_s"], abs=0.01)
    assert f["max_delta_action"] is not None and f["max_delta_action"] < SMOOTH_RAD, f
    manifest = manifest_of(out)
    assert manifest["episodes"] and manifest["episodes"][0]["recovery"]["kind"] == "dagger"
    NUMBERS["drawer_dagger"] = {"takeover_t": f["takeover_t"], "max_delta_action": round(f["max_delta_action"], 4), "expert_passed": rec["passed"],
                                "failed_phase": result["failed_phase"], "contract_bags": bag_contract(out, "no-images")}


def test_the_drawer_coupled_run_records_takeovers_and_phase_episodes(run, drawer):
    out, rec_out = run.dir / "drawer_coupled", run.dir / "drawer_coupled" / "recovery"
    recovery = derive(run, "specs/recovery/drawer_so101xl_coupled.yaml", "drawer_coupled_recovery", **{"from": {"eval": rel(out)}},
                      phases={"quota": {"approach": 2, "grasp": 0, "open": 1}}, live={"triggers": [TRIGGERS["phase_time"], TRIGGERS["stall"]],
                                                                                  "quota": {"approach": 3, "grasp": 3, "open": 3}}, out=rel(rec_out))
    ev = drawer_eval(run, drawer, "drawer_coupled", {"plugin": "noisy_expert", "sigma": 0.03}, 2, recovery={"spec": recovery, "enabled": True})
    t0 = time.monotonic()
    assert eval_cli.main([ev, "all", "--headless"], root=REPO) in (0, 1)
    rows = list(all_tries(out))
    assert len(rows) == 2 and not [r for *_, r, _ in rows if r.get("error")]
    live = [f for *_, f in rows if f and f["takeover_t"] is not None]
    assert live and {f["trigger"] for f in live} <= {"phase_time", "stall"}, "no trigger fired"
    numbers = recovery_numbers(rec_out)
    assert "error" not in numbers["jobs"], numbers
    manifest = manifest_of(rec_out)
    assert manifest["episodes"] and {e["recovery"]["kind"] for e in manifest["episodes"]} <= {"phase", "dagger"}
    NUMBERS["drawer_coupled"] = {"wall_s": round(time.monotonic() - t0), "takeovers": Counter(f["trigger"] for f in live), "phase": numbers,
                                 "manifest_entries": len(manifest["episodes"]), "contract_bags": bag_contract(rec_out, "no-images")}
