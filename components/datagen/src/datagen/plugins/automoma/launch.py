"""Host side of the plan run: `docker run` of the datagen image (cuRobo + AutoMoMa, GPU) on one scene instance (DATAGEN-v2.md 3: the
planner never runs in the sim image, which has no torch). The repo is mounted read-only, the scene's folder read-write, the
cuRobo build and AutoMoMa's source beside them. One run at a time per process: cuRobo wants the GPU's memory."""
from __future__ import annotations

import os
import subprocess
import threading
import uuid
from pathlib import Path

from revolve_common.sim_session import REPO, Docker

IMAGE = os.environ.get("REVOLVE_DATAGEN_IMAGE", "revolve-datagen:latest")
CUROBO = REPO / "ThirdParty" / "curobo" / "install"
AUTOMOMA = REPO / "ThirdParty" / "automoma"
PYTHONPATH = "/repo/common:/repo/components/datagen/src:/opt/curobo/install:/opt/automoma"
_gpu = threading.Lock()


class PlanError(RuntimeError):
    """The plan container could not run to the end."""


def argv(name: str, instance: Path, task: str, robot: str, folder: Path, repo: Path = REPO) -> list[str]:
    """The `docker run` command: `folder` (the scene's, under the repo) is mounted writable at its repo path."""
    rel = Path(folder).resolve().relative_to(repo)
    cache = repo / "cache"
    cache.mkdir(exist_ok=True)
    inst = Path("/repo") / Path(instance).resolve().relative_to(repo)
    return ["docker", "run", "--rm", "--name", name, "--gpus", "all", "--shm-size", "2g", "-u", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/cache", "-e", "REPO=/repo", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", f"PYTHONPATH={PYTHONPATH}",
            "-v", f"{repo}:/repo:ro", "-v", f"{Path(folder).resolve()}:/repo/{rel}", "-v", f"{cache}:/cache",
            "-v", f"{CUROBO}:/opt/curobo/install:ro", "-v", f"{AUTOMOMA}:/opt/automoma:ro", "-w", "/repo", IMAGE,
            "python", "-m", "datagen.plugins.automoma.plan", "--instance", str(inst), "--task", task, "--robot", robot,
            "--out", f"/repo/{rel}"]


def run(prefix: str, instance: Path, task: str, robot: str, folder: Path, log: Path, timeout_s: float) -> int:
    """Run the plan container (named `<prefix>-plan-<id>`, so an interrupt's `stop_sims(prefix)` finds it) -> its exit code
    (0 planned, 3 no plan); PlanError when it failed otherwise or ran over `timeout_s`."""
    name = f"{prefix}-plan-{uuid.uuid4().hex[:6]}"
    log.parent.mkdir(parents=True, exist_ok=True)
    with _gpu, open(log, "w") as f:
        try:
            rc = subprocess.run(argv(name, instance, task, robot, folder), stdout=f, stderr=subprocess.STDOUT,
                                timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            Docker().rm(name)
            raise PlanError(f"the plan ran over {timeout_s:.0f} s; see {log}") from None
    if rc not in (0, 3):
        raise PlanError(f"the plan container exited with {rc}; see {log}")
    return rc
