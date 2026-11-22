"""Back-compat of the old evaluation CLIs (no docker): evaluate.evaluate's eval_summary.json against a golden captured from
d7a4cc2 (before the evaluation component), the manager's eval step argv and its readers of the summary."""
import json
import sys
from pathlib import Path

import pytest

COMP = Path(__file__).resolve().parents[1]
REPO = COMP.parents[1]
sys.path[:0] = [str(COMP), str(COMP / "src" / "evaluation"), str(REPO / "components" / "manager" / "src")]
import evaluate as E  # noqa: E402
import eval_progress as progress  # noqa: E402
import starts as S  # noqa: E402
from manager import plan as P  # noqa: E402

from revolve_common import specs  # noqa: E402
from test_evaluation_unit import TASK, fake_runner, tmp  # noqa: E402,F401  (tmp: a fresh dir under runs/)

GOLDEN = Path(__file__).resolve().parent / "golden" / "eval_summary_d7a4cc2.json"
VOLATILE = ("updated_at", "eval_wall_s")
SCRIPT = {("5_r0", 0): "grasp", ("5_r0/retry_1", 0): None, ("6_r0", 0): "open", ("6_r0/retry_1", 0): "open",
          ("6_r0/retry_2", 0): "open", ("7_r0", 0): "global", ("8_r0", 0): "ERR", ("9_r0", 0): "ERR", ("9_r0", 1): "ERR"}


def scenario(tmp, monkeypatch, capsys):
    """evaluate.evaluate on 5 seeds x 2 rollouts, retries 2, a stub runner -> (eval_summary.json as text with `tmp` masked, last PROGRESS)."""
    monkeypatch.setattr(E, "gpu_mib", lambda: None)
    monkeypatch.setattr(E, "RTF_FILE", tmp / "eval_rtf.json")
    start = dict(dir=str((tmp / "s").relative_to(REPO)))
    monkeypatch.setattr(S, "resolve", lambda *a, **k: {s: start for s in range(5, 10)})
    run, _ = fake_runner(SCRIPT)
    E.evaluate(TASK, None, range(5, 10), tmp / "eval", "heldout", tmp / "starts", runner=run, rollouts_per_seed=2,
               retries=2, parallel=2, rtf=0.2, say=lambda m: None)
    s = json.loads((tmp / "eval" / "heldout" / "eval_summary.json").read_text())
    for k in VOLATILE:
        s.pop(k)
    text = json.dumps(s, indent=1, sort_keys=True).replace(str(tmp.relative_to(REPO)), "<TMP>")
    last = [x for x in capsys.readouterr().out.splitlines() if x.startswith("PROGRESS ")][-1]
    return text, progress.parse(last)


def test_eval_summary_equals_the_golden(tmp, monkeypatch, capsys):
    text, last = scenario(tmp, monkeypatch, capsys)
    assert json.loads(text) == json.loads(GOLDEN.read_text())
    assert last["rollouts"] == last["rollouts_done"] == 10 and last["passed"] == 7 and last["first_try_passed"] == 6


def test_manager_argv_runs_the_old_cli(monkeypatch):
    spec = REPO / "specs" / "experiments" / "fridge_mini.yaml"
    r = specs.load_experiment(spec)
    paths = P.Paths(Path("/x/runs/fridge_mini"))
    params = P.step_params(P.new_state("x", spec), 0, "eval_heldout", r, paths)
    assert params["script"] == "components/evaluation/evaluate.py"
    got = {}
    monkeypatch.setattr(E, "evaluate", lambda *a, **kw: got.update(args=a, kw=kw) or dict(errors=0))
    assert E.main(params["argv"]) == 0
    assert got["kw"] == dict(policy="smolvla", rollouts_per_seed=r.experiment.eval.rollouts_per_seed,
                             retries=r.experiment.eval.retries, parallel=None, sources=[], budget_s=None, wait_s=None,
                             gui=False, video=False, runner_params={})
    assert got["args"][3:] == (str(paths.eval(0)), "heldout", str(paths.starts))


def test_manager_reads_the_golden_summary(tmp_path):
    s = json.loads(GOLDEN.read_text())
    (tmp_path / "eval_summary.json").write_text(json.dumps(s))
    out, err = P.collect("eval_heldout", dict(watch=str(tmp_path)), None)
    assert err is None and out["all_pass"] is False and out["rollouts"] == 10 and out["first_try_passed"] == 6
    assert out["all_pass_with_retries"] is False
    assert {f["failed_phase"] for f in out["failures"]} == {"grasp", "open", "global"}


def test_runner_params_are_opt_in(tmp, monkeypatch):
    run, calls = fake_runner({})
    monkeypatch.setattr(E, "gpu_mib", lambda: None)
    monkeypatch.setattr(E, "RTF_FILE", tmp / "eval_rtf.json")
    monkeypatch.setattr(S, "resolve", lambda *a, **k: {5: dict(dir=str((tmp / "s").relative_to(REPO)))})
    given = dict(actions_per_chunk=21, chunk_size_threshold=0.0)
    for kw in ({}, dict(runner_params=given)):
        s = E.evaluate(TASK, None, [5], tmp / "eval", "heldout" + str(len(kw)), tmp / "starts", runner=run, parallel=1,
                       rtf=0.2, say=lambda m: None, **kw)
    assert "runner_params" not in calls[0]["budget"] and "runner_params" not in json.loads(GOLDEN.read_text())
    assert calls[1]["budget"]["runner_params"] == given and s["runner_params"] == given


def test_runner_params_flags_are_validated(monkeypatch):
    monkeypatch.setattr(E, "evaluate", lambda *a, **kw: dict(errors=0, kw=kw))
    base = ["--task", TASK, "--seeds", "5", "--out", "o", "--starts", "s", "--policy", "oracle"]
    assert E.main(base + ["--actions-per-chunk", "21", "--chunk-size-threshold", "0"]) == 0
    seen = []
    monkeypatch.setattr(E, "evaluate", lambda *a, **kw: seen.append(kw["runner_params"]) or dict(errors=0))
    E.main(base)
    E.main(base + ["--chunk-size-threshold", "0.5", "--aggregate-fn", "latest_only"])
    assert seen == [{}, dict(chunk_size_threshold=0.5, aggregate_fn_name="latest_only")]    # opt-in: nothing by default
    for bad in (["--actions-per-chunk", "0"], ["--chunk-size-threshold", "1.5"], ["--aggregate-fn", "median"]):
        with pytest.raises(SystemExit):
            E.main(base + bad)
