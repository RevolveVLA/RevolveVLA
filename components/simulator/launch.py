"""Host side: run one simulator job in revolve-simulator (docker SDK) under revolve_common.wrapper.

  job = start(["replay", "--plan", P, "--attempt", "0"], "runs/sim/replay_x")   # a container, detached
  res = finish(job)        # waits, saves container.log, removes the container -> dict (exit code, result.json)
  res = run([...], job_dir)                # both

Mounts: the repo read-only at /repo, runs/ read-write at /repo/runs (job dirs and generated models live
there), ThirdParty/gz_kura/install read-only at /opt/gz_ws/install (the path it was built at). Assets are
read in place through the repo mount (assets, git-ignored, licence-bound: never baked in).
No published ports: the wrapper's HTTP API is reachable on the container's IP on docker's bridge (or a
manager's network). The manager launches components the same way.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.types import DeviceRequest

from revolve_common import gpu as gpu_health
from revolve_common import job as J
from revolve_common import specs

REPO = Path(__file__).resolve().parents[2]
IMAGE = os.environ.get("REVOLVE_SIM_IMAGE", "revolve-simulator:latest")
INSTALL = REPO / "ThirdParty" / "gz_kura" / "install"
CLI = "/repo/components/simulator/src/sim/cli.py"
LABEL = "revolve.component"


def gui_kw():
    """(env, volumes) for a gz GUI client on the host's X display (the earlier gz demo's GUI mode, minus
    its `xhost +`: the container runs as the host uid with the host's XAUTHORITY cookie mounted read-only)."""
    display, xauth = os.environ.get("DISPLAY"), os.environ.get("XAUTHORITY")
    if not display or not xauth or not Path(xauth).is_file():
        raise SystemExit("--gui needs DISPLAY and XAUTHORITY (the host's X / Xwayland session)")
    env = {"DISPLAY": display, "XAUTHORITY": "/tmp/.Xauthority"}    # GLX env on the GPU: cli.py run_gz(gui=True)
    vols = {"/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "ro"}, xauth: {"bind": "/tmp/.Xauthority", "mode": "ro"}}
    return env, vols


def cpath(p) -> str:
    """Host path under the repo -> its container path."""
    p = Path(p).resolve()
    return "/repo/" + str(p.relative_to(REPO)) if p != REPO else "/repo"


@dataclass
class Job:
    container: object
    dir: Path
    timeout: float
    t0: float

    def ip(self):
        self.container.reload()
        ns = self.container.attrs["NetworkSettings"]
        return ns.get("IPAddress") or next(n["IPAddress"] for n in ns["Networks"].values() if n.get("IPAddress"))


def start(args, job_dir, gpu=False, timeout=1200.0, linger_s=0.0, env=None, network=None, cli=CLI, image=None,
          name=None, labels=None, gui=False) -> Job:
    """cli: the entrypoint script (another component's sim-side script, e.g. recovery's regen_sim.py);
    image: an image built FROM revolve-simulator (the rollout's revolve-simulator-ros); labels: extra labels;
    gui: the host X display mounts + env (gui_kw), for a command that starts a gz GUI client."""
    if not (INSTALL / "setup.bash").is_file():
        raise SystemExit("Kura not built: ./revolve build simulator")
    if gpu:
        gpu_health.launch_check()             # a GPU in a broken state: fail now, not at the sim watchdog
    job_dir = Path(job_dir) if Path(job_dir).is_absolute() else REPO / job_dir
    job_dir.mkdir(parents=True, exist_ok=True)
    for f in (J.STATUS, J.RESULT, J.LOG):
        (job_dir / f).unlink(missing_ok=True)
    cmd = ["python3", cli, *[cpath(a) if str(a).startswith(str(REPO) + "/") else str(a) for a in args]]
    jid = f"sim_{job_dir.name}_{uuid.uuid4().hex[:6]}"
    J.write_model(job_dir / J.JOB, J.Job(id=jid, component="simulator", command=cmd, params={"args": list(map(str, args))}))
    (REPO / "runs").mkdir(exist_ok=True)
    genv, gvols = gui_kw() if gui else ({}, {})
    c = docker.from_env().containers.run(
        image or IMAGE, ["python3", "-m", "revolve_common.wrapper", "--job-dir", cpath(job_dir), "--linger-s", str(linger_s)],
        detach=True, user=f"{os.getuid()}:{os.getgid()}", working_dir="/repo", name=name,
        labels={LABEL: "simulator", "revolve.job": jid, **(labels or {})},
        environment={"HOME": "/tmp", "REPO": "/repo", "PYTHONDONTWRITEBYTECODE": "1",
                     "PYTHONPATH": "/repo/common:/repo/components/simulator/src/sim:/opt/gz_ws/install/lib/python",
                     **specs.robot_env(), **genv, **(env or {})},
        volumes={str(REPO): {"bind": "/repo", "mode": "ro"}, str(REPO / "runs"): {"bind": "/repo/runs", "mode": "rw"},
                 str(INSTALL): {"bind": "/opt/gz_ws/install", "mode": "ro"}, **gvols},
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])] if gpu else None,
        network=network, shm_size="1g")
    return Job(c, job_dir, timeout, time.time())


def finish(job: Job) -> dict:
    c = job.container
    try:
        try:
            code = c.wait(timeout=max(1.0, job.timeout - (time.time() - job.t0)))["StatusCode"]
        except Exception:                      # timeout: the wrapper's SIGTERM path cancels the command
            c.stop(timeout=20)
            code = None
        (job.dir / "container.log").write_bytes(c.logs())
    finally:
        c.remove(force=True)
    return dict(exit=code, wall_s=round(time.time() - job.t0, 1), result=_read(job.dir / J.RESULT), dir=job.dir)


def run(args, job_dir, **kw) -> dict:
    return finish(start(args, job_dir, **kw))


def _read(p):
    import json
    return json.loads(p.read_text()) if p.is_file() else None
