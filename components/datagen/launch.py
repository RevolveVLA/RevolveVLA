"""Host side: run one datagen job in revolve-datagen (docker SDK) under revolve_common.wrapper, on the GPU.

  job = start(["plan", "--task", T, "--seed", "3", "--out", D], job_dir)   # a container, detached
  res = finish(job)      # the simulator launcher's finish: waits, container.log, removes -> dict
  res = run([...], job_dir)

Mounts as the simulator's launch.py (repo read-only at /repo, runs/ read-write), plus cuRobo's install
(ThirdParty/curobo/install, read-only at /opt/curobo/install), AutoMoMa's source (ThirdParty/automoma,
read-only at /opt/automoma: its editable install) and cache/ as HOME (warp/torch kernel caches persist,
as the legacy planner adapter). No published ports.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import uuid
from pathlib import Path

import docker
from docker.types import DeviceRequest

from revolve_common import gpu
from revolve_common import job as J
from revolve_common import specs

REPO = Path(__file__).resolve().parents[2]
IMAGE = os.environ.get("REVOLVE_DATAGEN_IMAGE", "revolve-datagen:latest")
INSTALL = REPO / "ThirdParty" / "curobo" / "install"
AUTOMOMA = REPO / "ThirdParty" / "automoma"
ENTRY = "/repo/components/datagen/src/datagen/entry.py"
_s = importlib.util.spec_from_file_location("sim_launch", REPO / "components" / "simulator" / "launch.py")
sim_launch = sys.modules["sim_launch"] = importlib.util.module_from_spec(_s)   # registered: its dataclass needs it
_s.loader.exec_module(sim_launch)
cpath, finish, Job, LABEL = sim_launch.cpath, sim_launch.finish, sim_launch.Job, sim_launch.LABEL


def start(args, job_dir, timeout=1800.0, linger_s=0.0, env=None, network=None, image=None, entry=ENTRY,
          component="datagen", parser="progress:parse", pythonpath=()) -> Job:
    """image / entry / component / parser / pythonpath (container paths put first): another component on
    this image (recovery: FROM revolve-datagen)."""
    if not (INSTALL / "BUILD_OK").is_file():
        raise SystemExit("cuRobo not built: ./revolve build datagen")
    gpu.launch_check()                        # a GPU in a broken state: fail now, not at the job's watchdog
    job_dir = Path(job_dir) if Path(job_dir).is_absolute() else REPO / job_dir
    job_dir.mkdir(parents=True, exist_ok=True)
    for f in (J.STATUS, J.RESULT, J.LOG):
        (job_dir / f).unlink(missing_ok=True)
    cmd = ["python", entry, *[cpath(a) if str(a).startswith(str(REPO) + "/") else str(a) for a in args]]
    jid = f"{component}_{job_dir.name}_{uuid.uuid4().hex[:6]}"
    J.write_model(job_dir / J.JOB, J.Job(id=jid, component=component, command=cmd, params={"args": list(map(str, args))}))
    for d in ("cache", "runs"):
        (REPO / d).mkdir(exist_ok=True)
    c = docker.from_env().containers.run(
        image or IMAGE, ["python", "-m", "revolve_common.wrapper", "--job-dir", cpath(job_dir), "--parser", parser,
                "--linger-s", str(linger_s)],
        detach=True, user=f"{os.getuid()}:{os.getgid()}", working_dir="/repo", labels={LABEL: component, "revolve.job": jid},
        environment={"HOME": "/cache", "REPO": "/repo", "PYTHONDONTWRITEBYTECODE": "1",
                     "PYTHONPATH": "".join(f"{p}:" for p in pythonpath) + "/repo/common:/repo/components/datagen/src/datagen:/repo/components/simulator/src/sim:"
                                   "/opt/curobo/install:/opt/automoma", **specs.robot_env(), **(env or {})},
        volumes={str(REPO): {"bind": "/repo", "mode": "ro"}, str(REPO / "runs"): {"bind": "/repo/runs", "mode": "rw"},
                 str(REPO / "cache"): {"bind": "/cache", "mode": "rw"},
                 str(INSTALL): {"bind": "/opt/curobo/install", "mode": "ro"},
                 str(AUTOMOMA): {"bind": "/opt/automoma", "mode": "ro"}},
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
        network=network, shm_size="2g")
    return Job(c, job_dir, timeout, time.time())


def run(args, job_dir, **kw) -> dict:
    return finish(start(args, job_dir, **kw))
