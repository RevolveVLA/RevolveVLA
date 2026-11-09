"""Host side: run dataset jobs in revolve-dataset (docker SDK) under revolve_common.wrapper, CPU only.

  res = convert(task, episode_dirs, out, turn)          # bags -> LeRobot v3.0 dataset at out
  res = merge(task, [prev_dataset, new_dataset], out)   # datasets -> one dataset at out
  res = turn_dataset(task, turn_dir, turn, episode_dirs, prev_dataset=None)
        # turn_dir/dataset: the episodes converted (turn 0 / no previous dataset), else
        # turn_dir/dataset_new (this turn's episodes alone) merged after prev_dataset
  job = start(["convert", ...], job_dir); res = finish(job)

Mounts as the simulator's launch.py (repo read-only at /repo, runs/ read-write) + Rosetta's colcon install
(ThirdParty/rosetta/install, read-only at /opt/rosetta/install, sourced before the wrapper). Offline as the
earlier pipeline's dataset adapter: HF_HUB_OFFLINE, no token, HF_HOME inside the container;
docker's bridge network only for the wrapper's HTTP (no published ports). res: the simulator launcher's
finish() dict (exit, wall_s, result = result.json, dir).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import uuid
from pathlib import Path

import docker

from revolve_common import job as J

REPO = Path(__file__).resolve().parents[2]
IMAGE = os.environ.get("REVOLVE_DATASET_IMAGE", "revolve-dataset:latest")
INSTALL = REPO / "ThirdParty" / "rosetta" / "install"
ENTRY = "/repo/components/dataset/src/dataset/entry.py"
OFFLINE_ENV = {"HOME": "/tmp", "HF_HUB_OFFLINE": "1", "HF_HOME": "/tmp/hf", "HF_LEROBOT_HOME": "/tmp/hf/lerobot"}
_s = importlib.util.spec_from_file_location("sim_launch", REPO / "components" / "simulator" / "launch.py")
sim_launch = sys.modules.setdefault("sim_launch", importlib.util.module_from_spec(_s))
if not hasattr(sim_launch, "finish"):
    _s.loader.exec_module(sim_launch)
cpath, finish, Job, LABEL = sim_launch.cpath, sim_launch.finish, sim_launch.Job, sim_launch.LABEL


def _arg(a) -> str:
    """Host paths under the repo (absolute) -> container paths; anything else as is."""
    return cpath(a) if str(a).startswith(str(REPO) + "/") else str(a)


def start(args, job_dir, timeout=3600.0, linger_s=0.0, env=None, network=None) -> Job:
    if not (INSTALL / "BUILD_OK").is_file():
        raise SystemExit("Rosetta not built: ./revolve build dataset")
    job_dir = Path(job_dir) if Path(job_dir).is_absolute() else REPO / job_dir
    job_dir.mkdir(parents=True, exist_ok=True)
    for f in (J.STATUS, J.RESULT, J.LOG):
        (job_dir / f).unlink(missing_ok=True)
    cmd = ["python3", ENTRY, *map(_arg, args)]
    jid = f"dataset_{job_dir.name}_{uuid.uuid4().hex[:6]}"
    J.write_model(job_dir / J.JOB, J.Job(id=jid, component="dataset", command=cmd, params={"args": list(map(str, args))}))
    c = docker.from_env().containers.run(
        IMAGE, ["bash", "-c", 'source /opt/rosetta/install/setup.bash && exec python3 -m revolve_common.wrapper "$@"',
                "bash", "--job-dir", cpath(job_dir), "--parser", "progress:parse", "--linger-s", str(linger_s)],
        detach=True, user=f"{os.getuid()}:{os.getgid()}", working_dir="/repo", labels={LABEL: "dataset", "revolve.job": jid},
        environment={"REPO": "/repo", "PYTHONDONTWRITEBYTECODE": "1", **OFFLINE_ENV,
                     "PYTHONPATH": "/repo/common:/repo/components/dataset/src/dataset", **(env or {})},
        volumes={str(REPO): {"bind": "/repo", "mode": "ro"}, str(REPO / "runs"): {"bind": "/repo/runs", "mode": "rw"},
                 str(INSTALL): {"bind": "/opt/rosetta/install", "mode": "ro"}},
        network=network, shm_size="1g")
    return Job(c, job_dir, timeout, time.time())


def in_image(argv, timeout=600) -> tuple[int, str]:
    """A one-off command in revolve-dataset with the job mounts and Rosetta sourced, no wrapper (tests,
    parity scripts) -> (exit code, output)."""
    c = docker.from_env().containers.run(
        IMAGE, ["bash", "-c", 'source /opt/rosetta/install/setup.bash && exec "$@"', "bash", *map(_arg, argv)],
        detach=True, user=f"{os.getuid()}:{os.getgid()}", working_dir="/repo", network_mode="none",
        labels={LABEL: "dataset"}, environment={"REPO": "/repo", "PYTHONDONTWRITEBYTECODE": "1", **OFFLINE_ENV},
        volumes={str(REPO): {"bind": "/repo", "mode": "ro"}, str(REPO / "runs"): {"bind": "/repo/runs", "mode": "rw"},
                 str(INSTALL): {"bind": "/opt/rosetta/install", "mode": "ro"}})
    try:
        code = c.wait(timeout=timeout)["StatusCode"]
        return code, c.logs().decode()
    finally:
        c.remove(force=True)


def run(args, job_dir, **kw) -> dict:
    return finish(start(args, job_dir, **kw))


def _job_dir(out, kind):
    return Path(out).parent / "jobs" / f"dataset_{kind}_{Path(out).name}"


def convert(task, episodes, out, turn, job_dir=None, **kw) -> dict:
    return run(["convert", "--task", task, "--turn", str(turn), "--out", out, *episodes],
               job_dir or _job_dir(out, "convert"), **kw)


def merge(task, datasets, out, job_dir=None, **kw) -> dict:
    return run(["merge", "--task", task, "--out", out, *datasets], job_dir or _job_dir(out, "merge"), **kw)


def turn_dataset(task, turn_dir, turn, episodes, prev_dataset=None, **kw) -> dict:
    turn_dir = Path(turn_dir)
    if prev_dataset is None:
        return convert(task, episodes, turn_dir / "dataset", turn, **kw)
    res = convert(task, episodes, turn_dir / "dataset_new", turn, **kw)
    if res["exit"] != 0:
        return res
    return merge(task, [prev_dataset, turn_dir / "dataset_new"], turn_dir / "dataset", **kw)
