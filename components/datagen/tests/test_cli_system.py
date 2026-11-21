"""System test: `./revolve datagen` as a subprocess on the host venv with the fake plugins (CPU, no docker)."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

TESTS = Path(__file__).resolve().parent
REVOLVE = TESTS.parents[2] / "revolve"
SPEC = {
    "task": "t", "robot": "r", "out": "run", "generator": {"plugin": "fake_datagen", "fail_seeds": [0]},
    "recorder": {"plugin": "fake_datagen"}, "converter": {"plugin": "fake_datagen"},
    "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
    "splits": {"train": {"target": 3, "sampler": {"strategy": "random", "overrides": {"fake_datagen.y": 1.5}},
                         "pool": {"initial": [0, 4]}},
               "heldout": {"target": 1, "pool": {"initial": [10, 12]}, "record": False}},
}


def datagen(tmp_path, *args, spec=SPEC, env=None):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    env = os.environ | {"DATAGEN_PLUGIN_PATH": str(TESTS)} | (env or {})
    r = subprocess.run([sys.executable, str(REVOLVE), "datagen", str(p), *args], capture_output=True, text=True,
                       cwd=tmp_path, env=env)
    return r.returncode, r.stdout, r.stderr


def spec_with(tmp_path, **edits):
    return SPEC | {"out": str(tmp_path / "run")} | edits


def test_all_then_rerun(tmp_path):
    spec = spec_with(tmp_path)
    rc, out, err = datagen(tmp_path, "all", spec=spec)
    assert rc == 0, err
    assert "generate train: 3/3 accepted, 1 rejected" in out and "check target: PASS" in out
    m = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert m["splits"]["train"]["rejected"][0]["reason"] == "planner found no path"
    assert all(c["values"]["fake_datagen.y"] == 1.5 for c in m["splits"]["train"]["sampled"])      # override applied
    assert (tmp_path / "run" / "dataset" / "dataset.json").exists() and (tmp_path / "run" / "report.md").exists()
    assert datagen(tmp_path, "all", spec=spec)[0] == 0                                          # resume = no-op


def test_steps_split_and_dry_run(tmp_path):
    spec = spec_with(tmp_path)
    rc, out, _ = datagen(tmp_path, "--dry-run", spec=spec)
    assert rc == 0 and "[dry-run] generate train seed 1" in out and not (tmp_path / "run").exists()
    assert datagen(tmp_path, "sample", "--split", "heldout", spec=spec)[0] == 0
    assert set(json.loads((tmp_path / "run" / "manifest.json").read_text())["splits"]) == {"heldout"}
    rc, _, err = datagen(tmp_path, "generate", "--split", "nope", spec=spec)
    assert rc == 1 and "unknown split" in err


def test_check_fails_with_exit_1_before_the_work_is_done(tmp_path):
    spec = spec_with(tmp_path)
    assert datagen(tmp_path, "sample", spec=spec)[0] == 0
    rc, out, _ = datagen(tmp_path, "check", spec=spec)
    assert rc == 1 and "check target: FAIL train 0/3, heldout 0/1" in out


def test_errors_are_one_line_and_exit_1(tmp_path):
    bad = spec_with(tmp_path, splits={"a": {"target": 0, "pool": {"initial": [0, 3]}}})
    rc, _, err = datagen(tmp_path, "sample", spec=bad)
    assert rc == 1 and "splits.a.target" in err and "Traceback" not in err
    rc, _, err = datagen(tmp_path, "sample", spec=spec_with(tmp_path, generator={"plugin": "nope_plugin"}))
    assert rc == 0                                           # `sample` resolves no generator
    rc, _, err = datagen(tmp_path, "generate", spec=spec_with(tmp_path, generator={"plugin": "nope_plugin"}))
    assert rc == 1 and "no generator plugin 'nope_plugin'" in err
    rc, _, err = datagen(tmp_path, "sample", spec=spec_with(tmp_path, seed=7))
    assert rc == 1 and "another spec" in err


# ------------------------------------------------------------------ the pipeline end to end, through the CLI
PLAIN = {"generator": {"plugin": "fake_datagen"},
         "splits": {"train": {"target": 3, "pool": {"initial": [0, 4]}},
                    "heldout": {"target": 1, "pool": {"initial": [10, 12]}, "record": False}}}


def manifest(tmp_path):
    return json.loads((tmp_path / "run" / "manifest.json").read_text())


def used_seeds(tmp_path, split):
    return [e["config"]["seed"] for e in manifest(tmp_path)["splits"][split]["used"]]


@pytest.mark.parametrize("strategy", ["random", "stratified", "sobol", "lhs"])
def test_all_with_each_sampler_strategy(tmp_path, strategy):
    sampler = {"strategy": strategy, "overrides": {"fake_datagen.y": 1.5}} | (
        {"bins": {"fake_datagen.x": 3}} if strategy == "stratified" else {})
    splits = {"train": {"target": 3, "sampler": sampler, "pool": {"initial": [0, 5]}},
              "heldout": {"target": 1, "sampler": sampler, "pool": {"initial": [10, 12]}, "record": False}}
    rc, out, err = datagen(tmp_path, "all", spec=spec_with(tmp_path, splits=splits, **{"generator": PLAIN["generator"]}))
    assert rc == 0, err
    m = manifest(tmp_path)
    assert used_seeds(tmp_path, "train") == [0, 1, 2] and used_seeds(tmp_path, "heldout") == [10]
    xs = [c["values"]["fake_datagen.x"] for c in m["splits"]["train"]["sampled"]]
    assert len(set(xs)) == len(xs) and all(-0.5 <= x <= 0.5 for x in xs)
    assert {c["values"]["fake_datagen.y"] for c in m["splits"]["train"]["sampled"]} == {1.5}
    assert all(c["ok"] for c in m["checks"])


def test_a_killed_run_resumes_where_it_stopped(tmp_path):
    spec, log = spec_with(tmp_path, **PLAIN), tmp_path / "calls.log"
    rc, _, _ = datagen(tmp_path, "all", spec=spec, env={"FAKE_DATAGEN_KILL_AFTER": "2", "FAKE_DATAGEN_LOG": str(log)})
    assert rc != 0 and used_seeds(tmp_path, "train") == [0, 1]
    log.unlink()
    rc, _, err = datagen(tmp_path, "all", spec=spec, env={"FAKE_DATAGEN_LOG": str(log)})
    assert rc == 0, err
    assert [l for l in log.read_text().split("\n") if l.startswith("generate")] == ["generate 2", "generate 10"]
    assert used_seeds(tmp_path, "train") == [0, 1, 2] and used_seeds(tmp_path, "heldout") == [10]


def test_an_exhausted_pool_exits_1_but_keeps_manifest_and_report(tmp_path):
    train = {"target": 2, "sampler": {"strategy": "stratified", "bins": {"fake_datagen.x": 2}}, "pool": {"initial": [0, 3]}}
    spec = spec_with(tmp_path, generator={"plugin": "fake_datagen", "fail_seeds": [0, 1, 2, 3]},
                     splits={"train": train})
    rc, _, err = datagen(tmp_path, "all", spec=spec)
    assert rc == 1 and "datagen: 0/2 accepted, 4 rejected, no unused config left in cells" in err and "Traceback" not in err
    assert len(manifest(tmp_path)["splits"]["train"]["rejected"]) == 4
    report = (tmp_path / "run" / "report.md").read_text()
    assert "| train | 2 | 0 | 4 |" in report and report.count("| yes |") == 2         # both cells exhausted


def test_a_failed_check_exits_1_with_the_verdict_in_the_report(tmp_path):
    spec = spec_with(tmp_path, recorder={"plugin": "fake_datagen", "fail_seeds": [1]}, **PLAIN)
    rc, out, _ = datagen(tmp_path, "all", spec=spec)
    assert rc == 1 and "check target: FAIL" in out
    assert "| target | FAIL |" in (tmp_path / "run" / "report.md").read_text()
    assert not manifest(tmp_path)["checks"][0]["ok"]


def test_a_changed_spec_is_refused_on_an_existing_out(tmp_path):
    assert datagen(tmp_path, "all", spec=spec_with(tmp_path, **PLAIN))[0] == 0
    before = (tmp_path / "run" / "manifest.json").read_bytes()
    changed = spec_with(tmp_path, **PLAIN | {"splits": PLAIN["splits"] | {"heldout": PLAIN["splits"]["heldout"] | {"target": 2}}})
    rc, _, err = datagen(tmp_path, "all", spec=changed)
    assert rc == 1 and "another spec" in err
    assert (tmp_path / "run" / "manifest.json").read_bytes() == before


def test_a_too_close_realised_start_is_replaced_and_the_run_passes(tmp_path):
    realised = {0: -0.4, 1: 0.4, 20: -0.38, 21: 0.0, 22: 0.1}               # heldout 20 is 0.02 from train 0
    splits = {"train": {"target": 2, "pool": {"initial": [0, 3]}},
              "heldout": {"target": 2, "pool": {"initial": [20, 29]}, "record": False,
                          "min_distance": {"from": "train", "value": 0.1, "realised": True}}}
    spec = spec_with(tmp_path, generator={"plugin": "fake_datagen", "realised": realised}, splits=splits)
    rc, out, err = datagen(tmp_path, "all", spec=spec)
    assert rc == 0, out + err
    held = manifest(tmp_path)["splits"]["heldout"]
    assert held["rejected"] == [{"seed": 20, "step": "realised", "reason": "min_distance (realised)"}]
    assert used_seeds(tmp_path, "heldout") == [21, 22] and "check realised_min_distance: PASS" in out


# ------------------------------------------------------------------ runtime flags
def test_workers_and_gui_flags_override_the_run_block(tmp_path):
    slow = {"plugin": "fake_datagen", "delay": 0.2}
    spec = spec_with(tmp_path, **PLAIN | {"generator": slow, "run": {"workers": 1}})
    t0 = time.time()
    rc, out, err = datagen(tmp_path, "generate", "--workers", "3", spec=spec)
    assert rc == 0, err
    assert time.time() - t0 < 2.0 and used_seeds(tmp_path, "train") == [0, 1, 2]       # 4 seeds of 0.2 s, 3 at a time
    (tmp_path / "gui").mkdir()
    rc, out, _ = datagen(tmp_path / "gui", "sample", "--workers", "3", "--gui", spec=spec_with(tmp_path / "gui", **PLAIN))
    assert rc == 0 and "run.headless false: one GUI window, so workers 3 -> 1" in out
    rc, out, _ = datagen(tmp_path / "gui", "sample", "--gui", spec=spec_with(tmp_path / "gui", **PLAIN | {"run": {"workers": 2}}))
    assert "workers 2 -> 1" in out
    rc, out, _ = datagen(tmp_path / "gui", "sample", "--headless",
                         spec=spec_with(tmp_path / "gui", **PLAIN | {"run": {"workers": 2, "headless": False}}))
    assert rc == 0 and "GUI" not in out
    rc, _, err = datagen(tmp_path, "sample", "--workers", "0", spec=spec)
    assert rc != 0 and "--workers must be >= 1" in err


def test_the_run_block_never_refuses_an_existing_out(tmp_path):
    assert datagen(tmp_path, "all", spec=spec_with(tmp_path, **PLAIN))[0] == 0
    rc, _, err = datagen(tmp_path, "all", spec=spec_with(tmp_path, **PLAIN | {"run": {"workers": 4, "headless": False}}))
    assert rc == 0, err


# ------------------------------------------------------------------ signals
def interrupted(tmp_path, sig, extra=()):
    """Start a slow run (4 configs, 1.5 s each, 2 at a time), send `sig` once two are under way -> (rc, stderr)."""
    log = tmp_path / "calls.log"
    spec = spec_with(tmp_path, **PLAIN | {"generator": {"plugin": "fake_datagen", "delay": 1.5}, "run": {"workers": 2}})
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    env = os.environ | {"DATAGEN_PLUGIN_PATH": str(TESTS), "FAKE_DATAGEN_LOG": str(log)}
    p = subprocess.Popen([sys.executable, str(REVOLVE), "datagen", str(tmp_path / "spec.yaml"), "generate", *extra],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path, env=env)
    for _ in range(100):
        if log.exists() and len(log.read_text().split()) >= 4:
            break
        time.sleep(0.1)
    p.send_signal(sig)
    out, err = p.communicate(timeout=30)
    return p.returncode, err, spec


@pytest.mark.parametrize("sig,code", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_a_signal_stops_cleanly_keeps_the_manifest_and_the_run_resumes(tmp_path, sig, code):
    rc, err, spec = interrupted(tmp_path, sig)
    assert rc == code and f"interrupted by {sig.name}" in err and "Traceback" not in err
    m = manifest(tmp_path)                                        # valid JSON, nothing half-decided
    assert m["splits"]["train"]["used"] == [] and (tmp_path / "run" / "report.md").exists()
    assert not list((tmp_path / "run" / "train" / "episodes").glob("seed_*"))        # unfinished episodes are not kept
    rc, out, err = datagen(tmp_path, "generate", "--workers", "4", spec=spec)
    assert rc == 0, err
    assert used_seeds(tmp_path, "train") == [0, 1, 2] and used_seeds(tmp_path, "heldout") == [10]
