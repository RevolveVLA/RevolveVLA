"""The `automoma` generator on the real sim (GPU + docker; skipped unless DATAGEN_GPU=1): the AutoMoMa plan (cuRobo in the datagen
image) of a fridge scene instance streamed to the sim must pass the evaluator at the base scene's start, with a valid manifest and a
bag that follows the contract (frozen topics, cameras, 30 Hz action); then AUTOMOMA_N (5) randomised instances of the repo's
datagen spec with one try each: their pass rate and the plan and episode wall times are reported (runs/automoma_gpu_<ts>/numbers.json),
not gated. Needs `./revolve build simulator`, the datagen image (cuRobo built: ThirdParty/curobo/install) and the scene's assets."""
import json
import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

from datagen import cli, contract
from revolve_common.sim_session import ClientContainer, Docker

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("DATAGEN_GPU") != "1", reason="needs a GPU: DATAGEN_GPU=1")]

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "runs" / f"automoma_gpu_{time.strftime('%Y%m%d_%H%M%S')}"
N = int(os.environ.get("AUTOMOMA_N", "5"))


@pytest.fixture(autouse=True)
def no_containers_left():
    left = lambda: [n for n in Docker().names() if n.startswith(("dg-", "acc-"))]      # noqa: E731
    assert not left(), f"containers of another run: {left()}"
    yield
    assert not left(), f"containers left: {left()}"


def run(spec, out, ok_rc=(0,)):
    """The pipeline on `spec` -> (manifest.json parsed, wall time)."""
    t0 = time.time()
    rc = cli.main([str(spec), "all", "--headless", "--workers", "1"], root=REPO)
    assert rc in ok_rc, f"datagen exited {rc}: see {out}/logs"
    assert contract.check_manifest(out) == []
    return json.loads((out / "manifest.json").read_text()), round(time.time() - t0, 1)


def numbers(name, manifest, wall, out):
    tries = [t for p in sorted(out.glob("**/scene_*/datagen.json")) for t in json.loads(p.read_text())["tries"]]
    plans = [json.loads(p.read_text()) for p in sorted(out.glob("**/scene_*/plan.json"))]
    eps = manifest["episodes"]
    row = {"attempted": len(eps), "passed": sum(e["passed"] for e in eps), "wall_s": wall,
           "planned": sum(p["ok"] for p in plans), "no_plan": [p["reason"] for p in plans if not p["ok"]],
           "plan_s": [p["report"]["times"]["plan_s"] + p["report"]["times"]["setup_s"] for p in plans if p["ok"]],
           "try_wall_s": [t["wall_s"] for t in tries], "grasp": [p["report"]["grasp"] for p in plans if p["ok"]]}
    RUN.mkdir(parents=True, exist_ok=True)
    f = RUN / "numbers.json"
    f.write_text(json.dumps({**(json.loads(f.read_text()) if f.exists() else {}), name: row}, indent=1))
    print(name, json.dumps(row))
    return row


def drop_bags(out):
    for bag in out.glob("**/episode_*_bag"):
        shutil.rmtree(bag)


def test_the_base_scene_start_passes_and_follows_the_contract():
    out = RUN / "base"
    scene = out / "scene_0" / "scene.yaml"
    scene.parent.mkdir(parents=True)
    shutil.copy(REPO / "specs" / "scenes" / "kitchen_partnet" / "scene.yaml", scene)       # the plain start
    job = out / "job.yaml"
    job.write_text(yaml.safe_dump({"task": "open_fridge_door", "robot": "mobile_panda", "episodes": [{"seed": 0, "scene": str(scene)}],
                                   "episodes_per_scene": 1, "out": str(out),
                                   "generator": {"plugin": "automoma", "attempts": 2, "record": "images"}}))
    manifest, wall = run(job, out)
    row = numbers("base", manifest, wall, out)
    assert row["passed"] == 1 and row["planned"] == 1, row
    client = ClientContainer(f"acc-{os.getpid()}", 98, ["python3", "-m", "datagen.contract", "--out", str(out), "--record", "images"],
                             out, pythonpath=["/repo/components/datagen/src"])
    log = RUN / "contract.log"
    rc = client.run(log, timeout=600)
    result = json.loads([ln for ln in log.read_text().splitlines() if ln.startswith("RESULT ")][-1][len("RESULT "):])
    assert rc == 0 and result["problems"] == [], result["problems"]
    drop_bags(out)


def test_randomised_scene_instances_pass_rate_is_reported():
    out = RUN / "random"
    spec = yaml.safe_load((REPO / "specs" / "datagen" / "fridge_panda_v2.yaml").read_text())
    spec.update(out=str(out.relative_to(REPO)), generator={**spec["generator"], "plugin": "automoma", "attempts": 1,
                                                           "record": "no-images"})
    spec["splits"] = {"train": {"target": N, "sampler": {"strategy": "sobol"}, "pool": {"initial": [0, N - 1]}}}   # no replacement:
    # a failed instance stays a failure (the pass rate is of the first N instances); the run then ends rc 1
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "random.yaml").write_text(yaml.safe_dump(spec))
    manifest, wall = run(RUN / "random.yaml", out, ok_rc=(0, 1))
    row = numbers("random", manifest, wall, out)
    drop_bags(out)
    assert row["attempted"] + len(row["no_plan"]) >= 1
