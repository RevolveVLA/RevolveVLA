"""DATAGEN-v2.md 14 on the real sim (GPU + docker; skipped unless DATAGEN_GPU=1): `mock_manager.py` on smoke_v2 (one scene
instance, `moveit` expert, images) with 1 worker, then with 2 workers on two instances (no images). Each run must end with a
valid manifest.json (datagen.contract.check_manifest: schema, paths, instances validate), at least one accepted episode and no
container left; the 1-worker run's bag is also checked in a client container (frozen topics, JPEG cameras at 30 Hz, /tf_static,
30 Hz action series). Output runs/acceptance_gpu_<ts>/ (deleted afterwards). Needs `./revolve build simulator` and the assets."""
import json
import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

import mock_manager
from datagen import contract
from revolve_common.sim_session import ClientContainer, Docker

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("DATAGEN_GPU") != "1", reason="needs a GPU: DATAGEN_GPU=1")]

REPO = Path(__file__).resolve().parents[3]
PREFIXES = ("dg-", "acc-")


def leftovers():
    return [n for n in Docker().names() if n.startswith(PREFIXES)]


@pytest.fixture
def run_dir():
    assert not leftovers(), f"containers of another run: {leftovers()}"
    d = REPO / "runs" / f"acceptance_gpu_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    assert not leftovers(), f"containers left: {leftovers()}"


def mock_run(run_dir, workers, scenes, record):
    """smoke_v2 (start pinned to the base scene's: the expert's tuned case) with `scenes` instances and `record`, through mock_manager -> (out, manifest)."""
    spec = yaml.safe_load((REPO / "specs" / "datagen" / "smoke_v2.yaml").read_text())
    spec["out"] = str((run_dir / "out").relative_to(REPO))
    spec["generator"]["record"] = record
    spec["randomisers"]["robot_start"] = {"x_m": [-1.75, -1.75], "y_m": [1.1, 1.1], "yaw_rad": [1.5708, 1.5708]}   # the base start
    spec["splits"]["train"].update(target=scenes, pool={"initial": [0, scenes + 1], "replacement": [scenes + 2, scenes + 6]})
    (run_dir / "spec.yaml").write_text(yaml.safe_dump(spec))
    rc = mock_manager.main([str(run_dir / "spec.yaml"), "--workers", str(workers)])
    out = run_dir / "out"
    assert rc == 0, f"mock_manager exited {rc}: see {out}/logs"
    assert contract.check_manifest(out) == []
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["scenes"] == scenes and manifest["counts"]["passed"] >= 1, manifest["counts"]
    return out, manifest


def test_one_worker_manifest_and_bag_follow_the_contract(run_dir):
    out, manifest = mock_run(run_dir, 1, 1, "images")
    passed = [e for e in manifest["episodes"] if e["passed"]]
    assert passed and all(e["bag"] for e in manifest["episodes"])
    client = ClientContainer(f"acc-{os.getpid()}", 99, ["python3", "-m", "datagen.contract", "--out", str(out), "--record", "images"],
                             out, pythonpath=["/repo/components/datagen/src"])
    log = run_dir / "contract.log"
    rc = client.run(log, timeout=600)
    result = json.loads([ln for ln in log.read_text().splitlines() if ln.startswith("RESULT ")][-1][len("RESULT "):])
    assert rc == 0 and result["problems"] == [], result["problems"]
    assert len(result["bags"]) == len(manifest["episodes"])
    print("bags", json.dumps(result["bags"]))


def test_two_workers_merge_one_valid_manifest(run_dir):
    mock_run(run_dir, 2, 2, "no-images")
