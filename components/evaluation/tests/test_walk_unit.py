"""core/walk.py + core/results.py + the pipeline's rollout / report stages (CPU: fake rollout plugin, stub rollouts)."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from datagen.core.interfaces import Config

from evaluation import cli
from evaluation.core import results
from evaluation.core.interfaces import Built, EvalItem, Verdict
from evaluation.core.spec import RunnerSpec
from evaluation.core.walk import RetryPolicy, Walk
from evaluation.plugins.sets.heldout import HeldoutSet

import test_cli_unit as T
import test_sets_unit as S

TESTS = Path(__file__).resolve().parent
SRC = [str(TESTS.parents[2] / "components" / "datagen" / "src"), str(TESTS.parent / "src")]
SPEC = T.SPEC | {"rollout": {"plugin": "fake", "fail_seeds": [101, 102], "failed_phase": "grasp"}, "retries": 1,
                 "judge": {"criterion": "c0"}, "policy": {"plugin": "fake_eval"}, "rollouts": 2,
                 "sets": {"heldout": {"plugin": "fake_eval", "count": 3, "pool": {"initial": [100, 109]}}}}


@pytest.fixture(autouse=True)
def plugins(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(TESTS))
    monkeypatch.setenv("FAKE_EVAL_LOG", str(tmp_path / "calls.log"))
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(T.TRAIN))


def results_of(tmp_path, out="run"):
    return json.loads((tmp_path / out / "results.json").read_text())


# ------------------------------------------------------------------ retry policy
def state(tmp_path, phase):
    (tmp_path / "phase_states").mkdir(exist_ok=True)
    np.savez(tmp_path / "phase_states" / f"{phase}_start.npz", q=np.zeros(1))


@pytest.mark.parametrize("tries, rec, passed, saved, want", [
    (1, {"failed_phase": "grasp"}, False, "grasp", "grasp"),            # restorable + state
    (1, {"failed_phase": "grasp"}, False, None, None),                  # no saved state
    (1, {"failed_phase": "global"}, False, "global", None),             # not restorable
    (1, {"failed_phase": None, "error": "boom"}, False, None, None),    # stack error: no phase
    (1, {"failed_phase": None}, False, None, None),                     # judge failed, every phase passed
    (1, {"failed_phase": "grasp"}, True, "grasp", None),                # the judge passed
    (3, {"failed_phase": "grasp"}, False, "grasp", None),               # retries used up (retries 2)
    (2, {"failed_phase": "open"}, False, "open", "open"),
])
def test_retry_policy(tmp_path, tries, rec, passed, saved, want):
    if saved:
        state(tmp_path, saved)
    got = RetryPolicy(2).restore(tries, rec, passed, tmp_path)
    assert got == ((tmp_path, want) if want else None)


def test_retry_policy_falls_back_to_the_first_trys_state(tmp_path):
    first, retry = tmp_path / "first", tmp_path / "retry_1"
    first.mkdir(), retry.mkdir(), state(first, "open")
    assert RetryPolicy(2).restore(2, {"failed_phase": "open"}, False, retry, first) == (first, "open")
    assert RetryPolicy(2).restore(2, {"failed_phase": "open"}, False, retry) is None
    state(retry, "open")
    assert RetryPolicy(2).restore(2, {"failed_phase": "open"}, False, retry, first) == (retry, "open")


# ------------------------------------------------------------------ the walk on stubs
class Stub:
    """Rollout + policy + judge in one: `plan[(seed, n-th call for the seed)]` -> record fields; the calls are kept."""
    name, needs_ckpt = "stub", False

    def __init__(self, plan=None, delay=0.0):
        self.plan, self.delay, self.calls, self.n, self.restores = plan or {}, delay, [], {}, []

    def run(self, item, out, *, seed, restore=None, runner, policy, ckpt, gui=False):
        self.n[seed] = self.n.get(seed, 0) + 1
        self.calls.append((seed, out.name, restore and restore[1]))
        time.sleep(self.delay.get(seed, 0) if isinstance(self.delay, dict) else self.delay)
        out.mkdir(parents=True, exist_ok=True)
        rec = {"passed": True, "phases": {"grasp": "pass"}, "failed_phase": None, "error": None, "restore": restore and {}}
        rec.update(self.plan.get((seed, self.n[seed]), {}))
        if rec["failed_phase"] and restore is None:         # as the sim: a restored run saves no start of its restore phase
            state(out, rec["failed_phase"])
        self.restores.append(restore and restore[0].name)
        return rec

    def score(self, record, rollout_dir, task):
        return Verdict(record["passed"])


def walk(tmp_path, stub, retries=1, workers=1, abort=None, log=None):
    return Walk(stub, stub, stub, RetryPolicy(retries), RunnerSpec(), None, None, tmp_path, workers, False,
                log or (lambda m: None), abort)


def built(*seeds):
    return Built([EvalItem(Config("s", s), Path("start")) for s in seeds])


def run(w, b, rollouts=1, done=None):
    got = {}
    w.run("s", b, rollouts, done or {}, got.__setitem__)
    return got


FAIL = {"passed": False, "failed_phase": "grasp"}


def test_a_retry_runs_from_the_failed_phase_into_retry_dir(tmp_path):
    stub = Stub({(1, 1): FAIL})
    got = run(walk(tmp_path, stub), built(0, 1))
    assert stub.calls == [(0, "0_r0", None), (1, "1_r0", None), (1, "retry_1", "grasp")]
    t = got["1_r0"]["tries"]
    assert [x["restored_at"] for x in t] == [None, "grasp"] and [x["passed"] for x in t] == [False, True]
    assert got["1_r0"]["passed"] and t[1]["dir"] == str(tmp_path / "s" / "1_r0" / "retry_1")
    assert len(got["0_r0"]["tries"]) == 1


def test_retries_stop_at_the_maximum(tmp_path):
    stub = Stub({(1, n): FAIL for n in (1, 2, 3)})
    got = run(walk(tmp_path, stub, retries=2), built(1))
    assert [c[1] for c in stub.calls] == ["1_r0", "retry_1", "retry_2"] and not got["1_r0"]["passed"]
    assert stub.restores == [None, "1_r0", "1_r0"]                       # every retry restores the first try's start state
    assert run(walk(tmp_path / "b", Stub({(1, 1): FAIL}), retries=0), built(1))["1_r0"]["tries"][0]["passed"] is False


def test_no_retry_for_a_global_failure_or_a_judge_only_failure(tmp_path):
    stub = Stub({(0, 1): {"passed": False, "failed_phase": "global"}, (1, 1): {"passed": False}})
    got = run(walk(tmp_path, stub), built(0, 1))
    assert len(stub.calls) == 2 and not got["0_r0"]["passed"] and not got["1_r0"]["passed"]


def test_a_stack_error_is_run_once_more_then_recorded(tmp_path):
    err = {"passed": False, "phases": None, "error": "docker died"}
    stub = Stub({(0, 1): err, (1, 1): err, (1, 2): err})
    got = run(walk(tmp_path, stub), built(0, 1))
    assert [c[:2] for c in stub.calls] == [(0, "0_r0"), (0, "0_r0"), (1, "1_r0"), (1, "1_r0")]
    assert got["0_r0"]["passed"] and not got["1_r0"]["passed"] and len(got["1_r0"]["tries"]) == 1
    assert got["1_r0"]["tries"][0]["detail"] == {"error": "docker died"}


def test_a_finished_try_on_disk_is_kept(tmp_path):
    stub = Stub({(1, 1): FAIL})
    w = walk(tmp_path, stub)
    first = run(w, built(0, 1))
    for d in ("0_r0", "1_r0", "1_r0/retry_1"):                       # what rollout.run leaves
        (tmp_path / "s" / d / "rollout.json").write_text(json.dumps(
            {"passed": d != "1_r0", "phases": {"grasp": "pass"}, "failed_phase": None if d != "1_r0" else "grasp",
             "restore": {} if "retry" in d else None}))
    (tmp_path / "s" / "1_r0" / "retry_1" / "rollout.json").write_text(json.dumps(
        {"passed": True, "phases": {}, "failed_phase": None, "restore": {"phase": "grasp"}}))
    stub.calls.clear()
    again = run(w, built(0, 1))
    assert stub.calls == [] and again["1_r0"]["passed"] and len(again["1_r0"]["tries"]) == 2 and first.keys() == again.keys()


def test_done_items_are_not_run_again(tmp_path):
    stub = Stub()
    w = walk(tmp_path, stub)
    first = run(w, built(0, 1, 2))
    stub.calls.clear()
    got = run(w, built(0, 1, 2), done={k: first[k] for k in ("0_r0", "1_r0")})
    assert [c[0] for c in stub.calls] == [2] and list(got) == ["2_r0"]


def test_progress_lines_are_what_eval_progress_reads(tmp_path):
    lines = []
    run(walk(tmp_path, Stub({(1, 1): FAIL}), log=lines.append), built(0, 1))
    from evaluation import eval_progress
    progress = [json.loads(m[len("PROGRESS "):]) for m in lines if m.startswith("PROGRESS ")]
    assert progress[0] == {"set": "s", "rollouts": 2, "rollouts_done": 0, "passed": 0, "first_try_passed": 0, "running": ["0_r0"]}
    assert progress[-1] == {"set": "s", "rollouts": 2, "rollouts_done": 2, "passed": 2, "first_try_passed": 1, "running": []}
    assert any(eval_progress.ROLLOUT.search(m) for m in lines)
    assert any("1_r0: try 0: FAIL grasp, try 1: PASS" in m for m in lines)


def test_an_abort_calls_the_hook_and_starts_no_second_attempt(tmp_path):
    err = {"passed": False, "phases": None, "error": "killed"}
    stub, hooked = Stub({(1, 1): err}, delay={0: 0.05, 1: 0.4}), []

    def boom(key, entry):
        raise KeyboardInterrupt
    w = walk(tmp_path, stub, workers=2, abort=lambda: hooked.append(1))
    with pytest.raises(KeyboardInterrupt):
        w.run("s", built(0, 1), 1, {}, boom)
    assert hooked == [1] and sorted(c[:2] for c in stub.calls) == [(0, "0_r0"), (1, "1_r0")]     # no second attempt of the killed one


# ------------------------------------------------------------------ results + pipeline
def test_workers_do_not_change_results_json(tmp_path):
    def go(workers):
        out = f"run{workers}"
        T.evaluate(tmp_path, spec=SPEC | {"out": out, "rollout": {**SPEC["rollout"], "delay_s": 0.05}}, *(["--workers", str(workers)]))
        return json.loads((tmp_path / out / "results.json").read_text().replace(str(tmp_path / out), "OUT"))
    one, three = go(1), go(3)
    assert one == three and len(one["sets"]["heldout"]["rollouts"]) == 6


def test_results_and_report_content(tmp_path):
    assert T.evaluate(tmp_path, spec=SPEC) == 0
    r = results_of(tmp_path)["sets"]["heldout"]
    assert r["metrics"] == {"rollouts": 6, "first_try_passed": 2, "first_try_success": 2 / 6, "passed_with_retries": 6,
                            "success_with_retries": 1.0, "first_try_failures": {"grasp": 4}, "errors": 0,
                            "terminations": {"budget": 4, "success": 6}}
    assert [t["termination"] for t in r["rollouts"]["101_r1"]["tries"]] == ["budget", "success"]
    assert [t["restored_at"] for t in r["rollouts"]["101_r1"]["tries"]] == [None, "grasp"]
    assert r["rollouts"]["100_r0"]["tries"][0]["record"]["phases"]["grasp"] == "pass"
    assert [c["seed"] for c in r["configs"]] == [100, 101, 102]
    md = (tmp_path / "run" / "report.md").read_text()
    assert "| 6 | 2 (33%) | 6 (100%) | 0 |" in md and "| grasp | 4 |" in md and "| 101 |  |  | 0/2 | 2/2 |" in md
    assert "Judge c0. Checkpoint: none" in md


def test_a_changed_spec_hash_refuses_the_results(tmp_path):
    results.Results.open(tmp_path / "r.json", "aaa").save()
    with pytest.raises(results.ResultsError, match="another spec"):
        results.Results.open(tmp_path / "r.json", "bbb")


def test_unset_workers_use_the_rollout_plugins_default(tmp_path, monkeypatch):
    from evaluation.core.pipeline import Pipeline
    from evaluation.plugins.rollouts.fake import FakeRollout
    monkeypatch.setattr(FakeRollout, "default_workers", lambda self, policy: 3)
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(SPEC))
    spec = cli.load(str(tmp_path / "spec.yaml"), tmp_path)
    p = Pipeline(spec, cli.build_plugins(spec, ("rollout",)), tmp_path / "run", log=lambda m: None)
    assert p.rollout_workers() == 3 and p.workers == 1                      # the sample stage keeps its own default
    p.spec = cli.with_run(spec, SimpleNamespace(workers=2, gui=False, headless=False))
    assert p.rollout_workers() == 2
    p.gui = True
    assert p.rollout_workers() == 1


# ------------------------------------------------------------------ abort in the sample stage, SIGINT
def test_the_sample_stage_aborts_what_its_workers_started(tmp_path, monkeypatch):
    spec = S.make(tmp_path)
    monkeypatch.setattr(S.heldout, "instancer_for", lambda spec: S.SceneWriter())
    monkeypatch.setattr(S, "KILL_AFTER", [1])
    S.CALLS.clear()
    aborted = []
    with pytest.raises(KeyboardInterrupt):
        HeldoutSet(spec).build(spec.sets["heldout"], tmp_path / "run", 1, lambda: aborted.append(1))
    assert aborted == [1]


def test_sigint_stops_the_run_with_130_and_resumes(tmp_path):
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(SPEC | {"rollout": {"plugin": "fake", "delay_s": 2}, "retries": 0,
                                                              "rollouts": 1}))
    code = ("import sys; from pathlib import Path; sys.path[:0] = sys.argv[2:]; from evaluation import cli;"
            "sys.exit(cli.main([sys.argv[1], 'rollout'], Path(sys.argv[1]).parent))")
    env = dict(os.environ, EVAL_PLUGIN_PATH=str(TESTS), DATAGEN_PLUGIN_PATH=str(TESTS.parents[1] / "datagen" / "tests"),
               FAKE_EVAL_LOG=str(tmp_path / "calls.log"))
    p = subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "spec.yaml"), *SRC], env=env, cwd=tmp_path,
                         stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 30
    while not (tmp_path / "run" / "heldout" / "starts" / "seed_102").exists() and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.7)                                            # inside the first rollout
    p.send_signal(signal.SIGINT)
    _, err = p.communicate(timeout=30)
    assert p.returncode == 130 and "interrupted by SIGINT" in err
    assert not (tmp_path / "run" / "results.json").exists() or not results_of(tmp_path)["sets"]["heldout"].get("rollouts")
