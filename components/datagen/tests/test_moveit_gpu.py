"""The moveit expert on the real sim (GPU + docker; skipped unless DATAGEN_GPU=1): per task, one scene instance at the base
scene's start must pass the evaluator (a DATAGEN 8.2 job through `datagen.cli`: `revolve sim` + the client container +
the `moveit` generator, `--record no-images`), then MOVEIT_N (5) randomised instances of the repo's datagen spec with one
attempt each: their pass rate is reported (runs/moveit_gpu_<ts>/numbers.json), not gated. Bags are deleted after the
numbers are written. Needs `./revolve build simulator` and the scene's assets.

The resume cases (`moveit_resume_client.py`), per task, RESUME_N (3) episodes each, at least two in three must pass: the sim starts
at the expert's own pre-grasp (base + arm, found by a probe) and the expert resumes there without folding or driving; and
the expert drives 2 s of the scene's start, stops, and `resume()` finishes the task. Numbers (entry, max_delta_action) are printed."""
import json
import math
import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

from datagen import cli
from revolve_common.sim_session import Docker, DomainPool, SimSession, run_with_relaunch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("DATAGEN_GPU") != "1", reason="needs a GPU: DATAGEN_GPU=1")]

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "runs" / f"moveit_gpu_{time.strftime('%Y%m%d_%H%M%S')}"
N = int(os.environ.get("MOVEIT_N", "5"))
RESUME_N = int(os.environ.get("RESUME_N", "3"))
CLIENT, SRC = "/repo/components/datagen/tests/moveit_resume_client.py", "/repo/components/datagen/src"
TASKS = {"open_fridge_door": ("mobile_panda", "fridge_panda_v2"), "open_drawer": ("mobile_so101_xl", "drawer_so101xl_v2")}


def leftovers():
    return [n for n in Docker().names() if n.startswith("dg-")]


def run(spec, out, ok_rc=(0,)):
    """The pipeline on `spec` -> manifest.json (parsed) and the wall time."""
    t0 = time.time()
    rc = cli.main([str(spec), "all", "--headless", "--workers", "1"], root=REPO)
    assert rc in ok_rc, f"datagen exited {rc}: see {out}/logs"
    return json.loads((out / "manifest.json").read_text()), round(time.time() - t0, 1)


def numbers(name, task, manifest, wall, out):
    eps = manifest["episodes"]
    tries = [t for p in sorted(out.glob("**/scene_*/datagen.json")) for t in json.loads(p.read_text())["tries"]]
    res = [json.loads((out / e["result"]).read_text()) for e in eps]
    row = {"task": task, "attempted": len(eps), "passed": sum(e["passed"] for e in eps), "wall_s": wall,
           "try_wall_s": [t["wall_s"] for t in tries],
           "failures": [{"seed": e["seed"], "phase": r["failed_phase"], "reason": r["reason"], "duration_s": r["duration_s"]}
                        for e, r in zip(eps, res) if not e["passed"]]}
    RUN.mkdir(parents=True, exist_ok=True)
    f = RUN / "numbers.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**data, name: row}, indent=1))
    print(name, json.dumps(row))
    return row


def drop_bags(out):
    for bag in out.glob("**/episode_*_bag"):
        shutil.rmtree(bag)


@pytest.fixture(autouse=True)
def no_containers_left():
    assert not leftovers(), f"containers of another run: {leftovers()}"
    yield
    assert not leftovers(), f"containers left: {leftovers()}"


@pytest.mark.parametrize("task", TASKS)
def test_the_base_scene_start_passes(task):
    robot, _ = TASKS[task]
    out = RUN / f"base_{task}"
    scene = out / "scene_0" / "scene.yaml"
    scene.parent.mkdir(parents=True)
    shutil.copy(REPO / "specs" / "scenes" / "kitchen_partnet" / "scene.yaml", scene)       # the plain start
    job = out / "job.yaml"
    job.write_text(yaml.safe_dump({"task": task, "robot": robot, "episodes": [{"seed": 0, "scene": str(scene)}],
                                   "episodes_per_scene": 1, "out": str(out),
                                   "generator": {"plugin": "moveit", "attempts": 2, "record": "no-images"}}))
    manifest, wall = run(job, out)
    row = numbers(f"base_{task}", task, manifest, wall, out)
    drop_bags(out)
    assert row["passed"] == 1, row


@pytest.mark.parametrize("task", TASKS)
def test_randomised_scene_instances_pass_rate_is_reported(task):
    robot, name = TASKS[task]
    out = RUN / f"random_{task}"
    spec = yaml.safe_load((REPO / "specs" / "datagen" / f"{name}.yaml").read_text())
    spec.update(out=str(out.relative_to(REPO)), generator={**spec["generator"], "attempts": 1, "record": "no-images"})
    spec["splits"] = {"train": {"target": N, "sampler": {"strategy": "sobol"}, "pool": {"initial": [0, N - 1]}}}   # no
    # replacement: a failed instance stays a failure (the pass rate is of the first N instances); the run then ends rc 1
    (RUN / f"random_{task}.yaml").parent.mkdir(parents=True, exist_ok=True)
    (RUN / f"random_{task}.yaml").write_text(yaml.safe_dump(spec))
    manifest, wall = run(RUN / f"random_{task}.yaml", out, ok_rc=(0, 1))
    row = numbers(f"random_{task}", task, manifest, wall, out)
    drop_bags(out)
    assert row["attempted"] >= 1


def sim_run(out, task, robot, mode):
    """The resume client on a sim of `out`/scene.yaml -> its RESULT (the rows of RESUME_N episodes)."""
    scene, log = out / "scene.yaml", out.parent / f"{out.name}.client.log"
    name, domain = DomainPool("dg-resume").lease()
    session = SimSession(scene, robot, task, out, name, domain, "no-images", log=out.parent / f"{out.name}.sim.log")

    def work(s):
        s.client(["python3", CLIENT, str(out), task, robot, str(scene), mode, str(RESUME_N)],
                 pythonpath=[SRC]).run(log, timeout=RESUME_N * 150 + 300, watch=s.check)
        line = [ln for ln in log.read_text().splitlines() if ln.startswith("RESULT ")][-1]
        return json.loads(line[len("RESULT "):])
    return run_with_relaunch(session, work)


def scene_at(out, robot_start=None):
    out.mkdir(parents=True)
    scene = yaml.safe_load((REPO / "specs" / "scenes" / "kitchen_partnet" / "scene.yaml").read_text())
    if robot_start:
        scene["robot_start"] = robot_start
    (out / "scene.yaml").write_text(yaml.safe_dump(scene))


@pytest.mark.parametrize("task", TASKS)
def test_the_expert_resumes_from_the_pre_grasp_and_after_a_takeover(task):
    robot, _ = TASKS[task]
    out = RUN / f"resume_{task}"
    scene_at(out / "takeover")
    takeover = sim_run(out / "takeover", task, robot, "takeover")
    start = takeover["pregrasp"]
    scene_at(out / "pregrasp", {"base": start["base"], "arm": start["arm"]})
    pregrasp = sim_run(out / "pregrasp", task, robot, "pregrasp")
    rows = {"takeover": takeover["episodes"], "pregrasp": pregrasp["episodes"]}
    RUN.mkdir(parents=True, exist_ok=True)
    print("RESUME", task, json.dumps({"pre_grasp_start": start, **rows}))
    shutil.rmtree(out)
    need = math.ceil(2 * RESUME_N / 3)
    for case, eps in rows.items():
        assert all(e["error"] is None for e in eps), eps
        assert sum(e["passed"] for e in eps) >= need, (case, eps)
    assert all(e["entry"] == "grasp" and "fold" not in e["whats"] and "drive_done" not in e["whats"] for e in rows["pregrasp"]), \
        rows["pregrasp"]
