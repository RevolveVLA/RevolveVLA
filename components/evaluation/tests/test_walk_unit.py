"""core/walk.py + core/results.py + the pipeline's rollout / report stages (CPU: fake rollout plugin, stub rollouts)."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
@pytest.mark.parametrize("tries, rec, passed, want", [
    (1, {"failed_phase": "grasp"}, False, True),                       # a failed episode: run again from the start
    (1, {"failed_phase": "global"}, False, True),                      # whatever phase failed
    (1, {"failed_phase": None, "error": "boom"}, False, False),        # stack error: not the policy's failure
    (1, {"failed_phase": "grasp"}, True, False),                       # the judge passed
    (2, {"failed_phase": "open"}, False, True),
    (3, {"failed_phase": "grasp"}, False, False),                      # retries used up (retries 2)
])
def test_retry_policy(tries, rec, passed, want):
    assert RetryPolicy(2).again(tries, rec, passed) is want
    assert RetryPolicy(0).again(1, {"failed_phase": "grasp"}, False) is False


# ------------------------------------------------------------------ the walk on stubs
class Stub:
    """Rollout + policy + judge in one: `plan[(seed, n-th call for the seed)]` -> record fields; the calls are kept."""
    name, needs_ckpt = "stub", False

    def __init__(self, plan=None, delay=0.0):
        self.plan, self.delay, self.calls, self.n, self.released, self.stopped = plan or {}, delay, [], {}, [], 0

    def run(self, item, out, *, seed, runner, policy, ckpt, gui=False):
        self.n[seed] = self.n.get(seed, 0) + 1
        self.calls.append((seed, out.name))
        time.sleep(self.delay.get(seed, 0) if isinstance(self.delay, dict) else self.delay)
        out.mkdir(parents=True, exist_ok=True)
        rec = {"passed": True, "phases": {"grasp": "passed"}, "failed_phase": None, "error": None}
        rec.update(self.plan.get((seed, self.n[seed]), {}))
        return rec

    def release(self, item):
        self.released.append(item.config.seed)

    def stop(self):
        self.stopped += 1

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


def test_a_retry_runs_again_from_the_start_into_the_retry_dir(tmp_path):
    stub = Stub({(1, 1): FAIL})
    got = run(walk(tmp_path, stub), built(0, 1))
    assert stub.calls == [(0, "0_r0"), (1, "1_r0"), (1, "retry_1")]
    t = got["1_r0"]["tries"]
    assert [x["passed"] for x in t] == [False, True] and "restored_at" not in t[0]
    assert got["1_r0"]["passed"] and t[1]["dir"] == str(tmp_path / "s" / "1_r0" / "retry_1")
    assert len(got["0_r0"]["tries"]) == 1


def test_retries_stop_at_the_maximum(tmp_path):
    stub = Stub({(1, n): FAIL for n in (1, 2, 3)})
    got = run(walk(tmp_path, stub, retries=2), built(1))
    assert [c[1] for c in stub.calls] == ["1_r0", "retry_1", "retry_2"] and not got["1_r0"]["passed"]
    assert run(walk(tmp_path / "b", Stub({(1, 1): FAIL}), retries=0), built(1))["1_r0"]["tries"][0]["passed"] is False


def test_a_config_runs_its_rollouts_in_order_then_is_released_once(tmp_path):
    stub = Stub()
    got = run(walk(tmp_path, stub), built(0, 1), rollouts=3)
    assert stub.calls == [(s, f"{s}_r{i}") for s in (0, 1) for i in range(3)] and stub.released == [0, 1]
    assert list(got) == [f"{s}_r{i}" for s in (0, 1) for i in range(3)]
    stub.calls.clear()
    again = run(walk(tmp_path, stub), built(0, 1), rollouts=3, done={k: got[k] for k in ("0_r0", "0_r1", "1_r0")})
    assert stub.calls == [(0, "0_r2"), (1, "1_r1"), (1, "1_r2")] and list(again) == ["0_r2", "1_r1", "1_r2"]


def test_the_sim_is_released_when_a_config_dies(tmp_path):
    class Boom(Stub):
        def run(self, item, out, **kw):
            raise RuntimeError("boom")
    stub = Boom()
    with pytest.raises(RuntimeError):
        run(walk(tmp_path, stub), built(0))
    assert stub.released == [0]


def test_a_failed_try_is_retried_whatever_its_phase(tmp_path):
    stub = Stub({(0, 1): {"passed": False, "failed_phase": "global"}, (1, 1): {"passed": False}})
    got = run(walk(tmp_path, stub), built(0, 1))
    assert [len(got[k]["tries"]) for k in ("0_r0", "1_r0")] == [2, 2] and got["0_r0"]["passed"]


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
    for d, ok in (("0_r0", True), ("1_r0", False), ("1_r0/retry_1", True)):                # what the rollout plugin leaves
        (tmp_path / "s" / d / "rollout.json").write_text(json.dumps(
            {"passed": ok, "phases": {"grasp": "passed"}, "failed_phase": None if ok else "grasp"}))
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


def test_progress_lines_carry_the_set_counters(tmp_path):
    lines = []
    run(walk(tmp_path, Stub({(1, 1): FAIL}), log=lines.append), built(0, 1))
    progress = [json.loads(m[len("PROGRESS "):]) for m in lines if m.startswith("PROGRESS ")]
    assert progress[0] == {"set": "s", "rollouts": 2, "rollouts_done": 0, "passed": 0, "first_try_passed": 0, "running": ["0_r0"]}
    assert progress[-1] == {"set": "s", "rollouts": 2, "rollouts_done": 2, "passed": 2, "first_try_passed": 1, "running": []}
    assert any("1_r0: try 0: FAIL grasp, try 1: PASS" in m for m in lines)


def test_an_abort_stops_the_rollout_calls_the_hook_and_starts_no_second_attempt(tmp_path):
    err = {"passed": False, "phases": None, "error": "killed"}
    stub, hooked = Stub({(1, 1): err}, delay={0: 0.05, 1: 0.4}), []

    def boom(key, entry):
        raise KeyboardInterrupt
    w = walk(tmp_path, stub, workers=2, abort=lambda: hooked.append(1))
    with pytest.raises(KeyboardInterrupt):
        w.run("s", built(0, 1), 1, {}, boom)
    assert hooked == [1] and stub.stopped == 1 and sorted(stub.calls) == [(0, "0_r0"), (1, "1_r0")]     # no second attempt of the killed one


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
                            "terminations": {"timeout_s": 4, "success": 6}}
    assert [t["termination"] for t in r["rollouts"]["101_r1"]["tries"]] == ["timeout_s", "success"]
    assert r["rollouts"]["100_r0"]["tries"][0]["record"]["phases"]["grasp"] == "passed"
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
