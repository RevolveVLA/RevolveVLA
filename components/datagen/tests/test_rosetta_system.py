"""GPU + docker system test of the rosetta plugins, skipped unless DATAGEN_GPU=1 (run on the AWS box, not on a shared GPU):
`./revolve datagen <spec> all` for one fridge seed: AutoMoMa plan + camera-off check (datagen, simulator images), the camera
replay + MCAP bag (recorder), then the LeRobot conversion by components/dataset (revolve-dataset image, Rosetta built).
Needs `./revolve build datagen simulator dataset` and the kitchen assets. Output: runs/datagen_tests/<stamp>_rosetta/."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("DATAGEN_GPU") != "1", reason="needs a GPU: DATAGEN_GPU=1")]

REPO = Path(__file__).resolve().parents[3]
SPEC = {"task": "open_fridge_door", "robot": "mobile_panda", "generator": {"plugin": "automoma", "attempts": 10},
        "recorder": {"plugin": "rosetta"}, "converter": {"plugin": "rosetta", "format": "lerobot"},
        "randomisers": {"robot_base": {"x_m": [-0.05, 0.05], "yaw_deg": [-5, 5]}},
        "splits": {"train": {"target": 1, "sampler": {"strategy": "random"}, "pool": {"initial": [0, 3]}, "record": True}}}


def test_generate_record_convert_one_episode():
    out = REPO / "runs" / "datagen_tests" / (time.strftime("%Y%m%d_%H%M%S") + "_rosetta")
    out.mkdir(parents=True)
    spec = out / "spec.yaml"
    spec.write_text(yaml.safe_dump(SPEC | {"out": str(out / "run")}))
    r = subprocess.run([sys.executable, str(REPO / "revolve"), "datagen", str(spec), "all"], capture_output=True, text=True,
                       cwd=REPO)
    (out / "stdout.log").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    m = json.loads((out / "run" / "manifest.json").read_text())
    [used] = m["splits"]["train"]["used"]
    episode = Path(used["record"]["paths"]["episode"])
    assert used["record"]["ok"] and (episode / "bag" / "metadata.yaml").is_file() and (episode / "episode.json").is_file()
    assert m["convert"]["result"]["ok"]
    summary = json.loads((out / "run" / "dataset" / "meta" / "revolve_dataset.json").read_text())
    assert summary["episodes"] == 1 and summary["frames"] > 0 and not summary["problems"]
    assert summary["labels"][0]["seed"] == used["config"]["seed"] and summary["labels"][0]["source"] == "demo"
