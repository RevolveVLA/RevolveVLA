"""cli.py + core/pipeline.py on fake plugins (CPU): dry-run, stages, resume, exit codes, imports."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from evaluation import cli

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[2]
TRAIN = {
    "task": "t", "robot": "r", "out": "train_run", "generator": {"plugin": "fake_datagen"},
    "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
    "splits": {"train": {"target": 3, "pool": {"initial": [0, 9]}}},
}
SPEC = {
    "train": "train.yaml", "out": "run", "ckpt": {"source": "fake_eval", "tag": "v1"},
    "policy": {"plugin": "fake_eval", "needs_ckpt": True}, "rollout": {"plugin": "fake_eval"},
    "judge": {"criterion": "fake_eval", "fail_seeds": [4]},
    "sets": {"heldout": {"plugin": "fake_eval", "count": 3, "pool": {"initial": [100, 109]}},
             "train": {"plugin": "fake_eval", "from": "train", "count": 2}},
    "rollouts": 2, "retries": 1,
}


@pytest.fixture(autouse=True)
def plugins(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(TESTS))
    monkeypatch.delenv("FAKE_EVAL_FAIL_CHECK", raising=False)
    monkeypatch.setenv("FAKE_EVAL_LOG", str(tmp_path / "calls.log"))
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(TRAIN))


def evaluate(tmp_path, *args, spec=SPEC):
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return cli.main([str(tmp_path / "spec.yaml"), *args], root=tmp_path)


def calls(tmp_path):
    p = tmp_path / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


def test_dry_run_prints_the_plan_and_touches_nothing(tmp_path, capsys):
    assert evaluate(tmp_path, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "[dry-run] set heldout (fake_eval): 3 held-out configs, random sampler, pool [100, 109]" in out
    assert "set train (fake_eval): 2 configs from training split 'train'" in out
    assert "ckpt: fake_eval {'tag': 'v1'} (nothing resolved)" in out
    assert "2 rollouts per config, retries 1: 10 first tries, up to 20 runs" in out and "workers: by free GPU memory" in out
    assert not (tmp_path / "run").exists() and calls(tmp_path) == []


def test_dry_run_needs_no_plugin(tmp_path, capsys):
    spec = SPEC | {"ckpt": {"source": "no_such_source"}, "judge": {"criterion": "nope"}}
    assert evaluate(tmp_path, "--dry-run", spec=spec) == 0


def test_all_runs_the_stages_and_resumes(tmp_path, capsys):
    assert evaluate(tmp_path) == 0
    out = capsys.readouterr().out
    assert "sample heldout: 3 configs" in out and "check heldout fake: PASS" in out
    r = json.loads((tmp_path / "run" / "results.json").read_text())
    assert r["ckpt"]["origin"] == "fake v1" and r["ckpt"]["path"].endswith("run/ckpt")
    roll = r["sets"]["heldout"]["rollouts"]
    assert sorted(roll) == [f"{s}_r{i}" for s in (100, 101, 102) for i in (0, 1)]
    assert roll["100_r0"]["passed"] and not roll["101_r0"]["passed"]
    assert (tmp_path / "run" / "heldout" / "100_r0").is_dir() and (tmp_path / "run" / "heldout" / "starts" / "seed_100").is_dir()
    assert (tmp_path / "run" / "report.md").read_text().count("## ") == 3 and "| 100 |" in (tmp_path / "run" / "report.md").read_text()
    assert len(calls(tmp_path)) == 10
    assert evaluate(tmp_path) == 0 and len(calls(tmp_path)) == 10                 # resume: nothing re-run


def test_a_killed_run_resumes_where_it_stopped(tmp_path, monkeypatch):
    import fake_eval
    n = []
    run = fake_eval.FakeRollout.run

    def dying(self, *a, **k):
        if len(n) == 3:
            raise KeyboardInterrupt
        n.append(1)
        return run(self, *a, **k)
    monkeypatch.setattr(fake_eval.FakeRollout, "run", dying)
    with pytest.raises(KeyboardInterrupt):
        evaluate(tmp_path, "rollout")
    monkeypatch.setattr(fake_eval.FakeRollout, "run", run)
    assert evaluate(tmp_path, "rollout") == 0
    assert len(calls(tmp_path)) == 10                                             # 3 + the 7 missing, none twice


def test_stages_and_set_filter(tmp_path):
    assert evaluate(tmp_path, "sample", "--set", "train") == 0
    assert (tmp_path / "run" / "train" / "starts").is_dir() and not (tmp_path / "run" / "heldout").exists()
    assert calls(tmp_path) == []
    assert evaluate(tmp_path, "report") == 0 and (tmp_path / "run" / "report.md").exists()
    assert evaluate(tmp_path, "rollout", "--set", "heldout") == 0 and len(calls(tmp_path)) == 6


def test_a_failed_check_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_EVAL_FAIL_CHECK", "1")
    assert evaluate(tmp_path, "check") == 1
    assert "check heldout fake: FAIL (scripted)" in capsys.readouterr().out


def test_unknown_set_and_missing_ckpt_are_errors(tmp_path, capsys):
    assert evaluate(tmp_path, "--set", "nope") == 1 and "unknown set ['nope']" in capsys.readouterr().err
    assert evaluate(tmp_path, "rollout", spec={k: v for k, v in SPEC.items() if k != "ckpt"}) == 1
    assert "needs a `ckpt:`" in capsys.readouterr().err
    assert evaluate(tmp_path, spec=SPEC | {"judge": {"criterion": "nope"}}) == 1
    assert "no judge plugin 'nope'" in capsys.readouterr().err


def test_a_changed_spec_refuses_the_existing_out(tmp_path, capsys):
    assert evaluate(tmp_path) == 0
    assert evaluate(tmp_path, spec=SPEC | {"rollouts": 3}) == 1
    assert "another spec" in capsys.readouterr().err
    assert evaluate(tmp_path, spec=SPEC | {"run": {"workers": 2}}) == 0              # `run` is not in the hash


def test_workers_and_gui_override_the_run_block(tmp_path, capsys):
    assert evaluate(tmp_path, "--dry-run", "--workers", "4") == 0
    assert "workers: 4, headless" in capsys.readouterr().out
    assert evaluate(tmp_path, "--dry-run", "--workers", "4", "--gui") == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith("workers: 1, gui")


def test_a_bad_spec_exits_2_naming_the_key(tmp_path, capsys):
    assert evaluate(tmp_path, spec=SPEC | {"retrys": 1}) == 2
    assert "spec.yaml.retrys: unknown key" in capsys.readouterr().err
    assert evaluate(tmp_path, spec=SPEC | {"train": "missing.yaml"}) == 2
    assert evaluate(tmp_path, "--dry-run", spec=SPEC | {"sets": {"h": {"count": 1, "pool": {"initial": [1, 2]}}}}) == 2
    assert "sets.h.pool" in capsys.readouterr().err


def test_importing_the_cli_pulls_no_docker_or_torch():
    code = ("import sys; sys.path[:0] = sys.argv[1:]; import evaluation.cli, evaluation.core.pipeline;"
            "bad = [m for m in ('docker', 'torch', 'huggingface_hub', 'rollout', 'launch', 'numpy', 'rclpy') if m in sys.modules];"
            "assert not bad, bad")
    src = [str(REPO / "components" / "datagen" / "src"), str(TESTS.parent / "src")]
    subprocess.run([sys.executable, "-c", code, *src], check=True)


def test_revolve_evaluate_dry_run_on_the_repo_spec():
    r = subprocess.run([sys.executable, str(REPO / "revolve"), "evaluate", "specs/eval/fridge_base50_30k.yaml", "--dry-run"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert "[dry-run] set heldout" in r.stdout and "[dry-run] ckpt: local" in r.stdout
    bad = subprocess.run([sys.executable, str(REPO / "revolve"), "evaluate", "--dry-run", "/dev/stdin"],
                         input=yaml.safe_dump({"train": "specs/datagen/fridge_base50.yaml", "out": "x", "sets": {}, "typo": 1}),
                         capture_output=True, text=True, cwd=REPO)
    assert bad.returncode == 2 and "typo: unknown key" in bad.stderr
