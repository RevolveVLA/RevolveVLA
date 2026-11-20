"""System test: `./revolve datagen` as a subprocess on the host venv with the fake plugins (CPU, no docker)."""
import json
import os
import subprocess
import sys
from pathlib import Path

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


def datagen(tmp_path, *args, spec=SPEC):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec))
    env = os.environ | {"DATAGEN_PLUGIN_PATH": str(TESTS)}
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
    assert rc == 1 and "check target: FAIL heldout 0/1, train 0/3" in out


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
