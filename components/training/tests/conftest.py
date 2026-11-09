"""Training tests. test_training_unit.py: progress parser, retrain-mode plans, the gpu_machine and modal backends
with fake ssh / a fake `modal` module (no docker). test_training.py: real jobs in revolve-training.

Fixtures are not committed binaries: the dataset component's cached episodes (components/dataset/tests: the
fridge seed 0 demo in runs/test_cache/recovery/datagen/, its recovered open suffix in
runs/test_cache/dataset/recovery/; `./revolve test dataset` makes them) converted by components/dataset/launch.py turn_dataset into
runs/test_cache/training/turn_0/dataset (demo) and turn_1/dataset (demo + recovery, labels turn 0 / 1) on the first
run, reused after (delete that dir to regenerate). Outputs: runs/training_tests/<stamp>/; numbers.json sums up.
The first local train downloads lerobot/smolvla_base (public, no token) into cache/hf.

Needs: `./revolve build training` and `./revolve build dataset`; the gpu_machine test passwordless ssh access to
$REVOLVE_GPU_MACHINE_HOST (user@host; unset or empty skips that test).
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

COMP = Path(__file__).resolve().parents[1]
REPO = COMP.parents[1]
sys.path[:0] = [str(COMP), str(COMP / "src" / "training")]
CACHE = REPO / "runs" / "test_cache" / "training"
RUN = REPO / "runs" / "training_tests" / time.strftime("%Y%m%d_%H%M%S")
TASK = "open_fridge_door"
GPU_MACHINE = os.environ.get("REVOLVE_GPU_MACHINE_HOST", "")


def load(p):
    return json.loads(Path(p).read_text())


def _module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def preflight():
    import launch as L
    if subprocess.run(["docker", "image", "inspect", L.IMAGE], capture_output=True).returncode:
        pytest.fail(f"not built: {L.IMAGE} (./revolve build training)", pytrace=False)


def make_datasets():
    """{"turn_0": demo-only dataset, "turn_1": demo + recovery merged} (host paths), cached in CACHE."""
    preflight()
    out = dict(turn_0=CACHE / "turn_0" / "dataset", turn_1=CACHE / "turn_1" / "dataset")
    if all((d / "meta" / "info.json").is_file() for d in out.values()):
        return out
    demo = REPO / "runs" / "test_cache" / "recovery" / "datagen" / TASK / "episodes" / "seed_0_a0"
    rec = REPO / "runs" / "test_cache" / "dataset" / "recovery" / "seed_0_a0_from_open" / "episodes" / "ep_0"
    if not all((e / "bag" / "metadata.yaml").is_file() for e in (demo, rec)):
        pytest.fail(f"no cached episodes {demo} / {rec}: run ./revolve test dataset first", pytrace=False)
    D = _module("dataset_launch", COMP.parent / "dataset" / "launch.py")
    r0 = D.turn_dataset(TASK, CACHE / "turn_0", 0, [demo])
    assert r0["exit"] == 0, r0
    r1 = D.turn_dataset(TASK, CACHE / "turn_1", 1, [rec], prev_dataset=out["turn_0"])
    assert r1["exit"] == 0, r1
    return out


@pytest.fixture(scope="session")
def datasets():
    return make_datasets()


@pytest.fixture(scope="session")
def numbers():
    d = {}
    yield d
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "numbers.json").write_text(json.dumps(d, indent=1, default=str))
