"""System test: `./revolve datagen <spec> sample|generate` as a subprocess on the host venv (CPU, fake generator): scene
instances for open_fridge_door / mobile_panda (the repo's own specs), each validated by `./revolve validate`."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

TESTS = Path(__file__).resolve().parent
REVOLVE = TESTS.parents[2] / "revolve"
ASSETS = TESTS.parents[2] / "assets" / "generated_partnet" / "fridge" / "model.sdf"
SPEC = {
    "task": "open_fridge_door", "robot": "mobile_panda", "generator": {"plugin": "fake_datagen"},
    "randomisers": {"robot_start": {"x_m": [-1.85, -1.65], "y_m": [1.0, 1.2], "yaw_rad": [1.47, 1.67],
                                    "arm": {"choices": ["folded", "ready"]}}},
    "splits": {"train": {"target": 3, "pool": {"initial": [0, 5]}}},
}


def run(tmp_path, *args, spec=SPEC):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec | {"out": str(tmp_path / "run")}, sort_keys=False))
    env = os.environ | {"DATAGEN_PLUGIN_PATH": str(TESTS)}
    return subprocess.run([sys.executable, str(REVOLVE), *args[:1], str(p), *args[1:]] if args[0] == "datagen" else
                          [sys.executable, str(REVOLVE), *args], capture_output=True, text=True, cwd=tmp_path, env=env)


def instances(tmp_path):
    return sorted((tmp_path / "run" / "train").glob("scene_*/scene.yaml"))


def test_sample_writes_instances_that_validate(tmp_path):
    r = run(tmp_path, "datagen", "sample")
    assert r.returncode == 0, r.stderr
    found = instances(tmp_path)
    assert [p.parent.name for p in found] == [f"scene_{i}" for i in range(6)]
    for p in found:
        v = run(tmp_path, "validate", "--no-assets", str(p))
        assert v.returncode == 0 and "OK" in v.stdout and "scene kitchen_partnet" in v.stdout, v.stdout + v.stderr
        start = yaml.safe_load(p.read_text())["robot_start"]
        assert -1.85 <= start["base"][0] <= -1.65 and start["arm"] in ("folded", "ready")
    assert run(tmp_path, "datagen", "sample").returncode == 0          # resume: same instances
    assert instances(tmp_path) == found


def test_old_randomisers_are_an_error_naming_robot_start(tmp_path):
    r = run(tmp_path, "datagen", "sample", spec=SPEC | {"randomisers": {"robot_base": {"x_m": [-0.1, 0.1]}}})
    assert r.returncode == 1 and "randomisers.robot_base: removed" in r.stderr and "robot_start" in r.stderr
    assert "Traceback" not in r.stderr


def test_a_start_outside_the_hard_limits_is_one_line(tmp_path):
    r = run(tmp_path, "datagen", "sample", spec=SPEC | {"randomisers": {"robot_start": {"x_m": [-1.85, 9.0]}}})
    assert r.returncode == 1 and "randomisers.robot_start.x_m" in r.stderr and "Traceback" not in r.stderr


def test_list_randomisers_shows_robot_start():
    r = subprocess.run([sys.executable, str(REVOLVE), "datagen", "--list-randomisers", "--task", "open_drawer",
                        "--robot", "mobile_so101_xl"], capture_output=True, text=True)
    assert r.returncode == 0 and "scene kitchen_partnet" in r.stdout and "robot_start.arm" in r.stdout


@pytest.mark.skipif(not ASSETS.is_file(), reason="needs ./revolve assets")
def test_generate_rejects_colliding_starts_and_replaces_them(tmp_path):
    spec = SPEC | {"validators": [{"plugin": "collision_free"}],
                   "randomisers": {"robot_start": {"x_m": [-3.2, -0.3], "y_m": [-0.3, 2.6], "yaw_rad": [-1.5, 4.7]}},
                   "splits": {"train": {"target": 4, "pool": {"initial": [0, 3], "replacement": [4, 60]}}}}
    r = run(tmp_path, "datagen", "generate", spec=spec)
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "run" / "manifest.json").read_text())["splits"]["train"]
    assert len(m["used"]) == 4 and m["rejected"] and all(e["reason"].startswith("collision") for e in m["rejected"])
