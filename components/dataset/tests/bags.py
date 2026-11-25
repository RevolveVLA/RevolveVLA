"""Real simulator episodes for the dataset tests (DATASET-v2.md section 11), recorded on the local GPU:

  manifest = record(root, robot, task, episodes, split="train", seed=0)
  python3 components/dataset/tests/bags.py --out DIR --robot R --task T --episodes N [--split S] [--seed K]

One sim (`revolve sim --scene <instance> --robot R --task T --no-autostart --record images --out
<root>/<split>/scene_<seed>`) and one client container (stream_client.py: a policy-like 30 Hz stream until each
result, a reset between episodes), then the sim removed. Writes <root>/manifest.fixture.json in datagen's manifest
schema, every recorded episode listed as passed: a TEST FIXTURE ONLY (the evaluator failed these episodes; a real
producer's manifest copies its `passed`). Episodes are about 20 s (the task's first phase timeout) and ~135 MB each."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
SIM_IMAGE = os.environ.get("REVOLVE_SIM_IMAGE", "revolve-simulator:latest")
FIXTURE = "manifest.fixture.json"
EXPERT = "stream_client_fixture"
NOTE = ("TEST FIXTURE: simulator episodes of a scripted streaming client; the evaluator failed them, and this manifest "
        "lists them as passed for the dataset component's tests only")


def scene_of_task(task) -> str:
    return yaml.safe_load((REPO / "specs" / "tasks" / task / "task.yaml").read_text())["scene"]


def record(root, robot, task, episodes, split="train", seed=0, timeout_s=None) -> Path:
    root = Path(root).resolve()
    run = root / split / f"scene_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    instance = run / "scene.yaml"
    instance.write_text(yaml.safe_dump(yaml.safe_load(
        (REPO / "specs" / "scenes" / scene_of_task(task) / "scene.yaml").read_text()), sort_keys=False))
    name, domain = f"dsrec-{os.getpid()}-{seed}", 150 + (os.getpid() + seed) % 80
    log = open(run / "sim.log", "w")
    sim = subprocess.Popen([str(REPO / "revolve"), "sim", "--scene", str(instance), "--robot", robot, "--task", task,
                            "--no-autostart", "--record", "images", "--out", str(run), "--name", name,
                            "--domain-id", str(domain)], stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    client = f"{name}-client"
    cmd = ["docker", "run", "--rm", "--name", client, "--network", "host", "-u", f"{os.getuid()}:{os.getgid()}",
           "-e", "HOME=/tmp", "-e", f"ROS_DOMAIN_ID={domain}", "-e", "FASTDDS_BUILTIN_TRANSPORTS=UDPv4",
           "-e", "PYTHONPATH=/repo/common", "-v", f"{REPO}:/repo:ro", "-v", f"{run}:{run}",
           "-v", f"{REPO}/ThirdParty/gz_kura/install:/opt/gz_ws/install:ro",
           "-v", f"{REPO}/ThirdParty/ros_gz_ws/install:/opt/ros_gz_ws/install:ro", SIM_IMAGE, "bash", "-c",
           f"python3 /repo/components/dataset/tests/stream_client.py --scene {instance} --robot {robot} --task {task} "
           f"--episodes {episodes} --seed {seed}"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s or 120 + 90 * episodes)
    finally:
        subprocess.run(["docker", "rm", "-f", client, name], capture_output=True)
        sim.wait(timeout=60)
        log.close()
    (run / "client.log").write_text(r.stdout + r.stderr)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode or not lines:
        raise RuntimeError(f"recording failed (client exit {r.returncode}): see {run / 'client.log'}, "
                           f"{run / 'sim.log'}")
    eps = json.loads(lines[-1][len("RESULT "):])["episodes"]
    rel = lambda p: str(Path(p).relative_to(root))      # noqa: E731
    for e in eps:
        f = Path(e["result"]["result_file"])
        e["fixture_result"] = f.with_name(f.stem + ".fixture.json")
        e["fixture_result"].write_text(json.dumps(dict(json.loads(f.read_text()), passed=True, fixture=NOTE), indent=1))
    entries = [dict(seed=seed, scene=rel(instance), robot=robot, task=task, episode=e["result"]["episode"],
                    result=rel(e["fixture_result"]), bag=rel(e["result"]["bag"]), passed=True,
                    datagen=dict(expert=EXPERT, attempt=i, wall_s=e["wall_s"])) for i, e in enumerate(eps)]
    m = root / FIXTURE
    old = json.loads(m.read_text())["episodes"] if m.is_file() else []
    old = [e for e in old if not e["scene"].startswith(f"{split}/scene_{seed}/")]
    doc = dict(task=task, robot=robot, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), fixture=NOTE,
               episodes=old + entries)
    doc["counts"] = dict(scenes=len({e["scene"] for e in doc["episodes"]}), attempted=len(doc["episodes"]),
                         passed=len(doc["episodes"]))
    m.write_text(json.dumps(doc, indent=1))
    print(f"recorded {len(entries)} episodes of {robot}/{task} into {run} in {time.time() - t0:.0f} s; {m}", flush=True)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    record(a.out, a.robot, a.task, a.episodes, a.split, a.seed)
    sys.exit(0)
