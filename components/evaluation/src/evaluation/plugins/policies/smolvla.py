"""SmolVLA on the new simulator: the checkpoint is served by a policy service container (the training image: LeRobot + torch + the
GPU, `evaluation.policy_service`, started at the first `client_args` and shared by every rollout of the run) and the policy client
in the sim's client container (`rollout_client --serve`, evaluation.served) builds the observation from the sim's topics and streams
the chunk's actions at 30 Hz (no blend: `actions_per_chunk` actions of each chunk are sent, then the next chunk is asked for).

Params (yaml `policy: {plugin: smolvla, ...}`): `actions_per_chunk` (executed per chunk; default the runner's `n_action_steps`, 20),
`image_size` (cameras resized to size x size; default none: the frames as the sim publishes them, 480, which the checkpoint was
trained on), `image` (the service's docker image, default $REVOLVE_TRAINING_IMAGE or revolve-training:latest), `device` (cuda),
`startup_s` (wall seconds for the service to load, 300). The robot must have a contract in evaluation.policy_contract."""
from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from revolve_common.sim_session import REPO

from ...core.errors import EvalError
from ...core.interfaces import Policy
from ...core.registry import REGISTRY
from ...core.sims import sim_prefix
from ...core.task import load_task
from ...policy_contract import CONTRACTS
from ...policy_wire import PolicyClient

PARAMS = ("actions_per_chunk", "image_size", "image", "device", "startup_s")
IMAGE = os.environ.get("REVOLVE_TRAINING_IMAGE", "revolve-training:latest")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Service:
    """The policy service container `name` serving `ckpt` (a path under the repo) on 127.0.0.1:`port`."""

    def __init__(self, name: str, ckpt: Path, state_dim: int, image: str = IMAGE, device: str = "cuda", repo: Path = REPO,
                 run=subprocess.run):
        self.name, self.ckpt, self.state_dim, self.image, self.device, self.repo, self.run = name, ckpt, state_dim, image, device, repo, run
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"

    def argv(self) -> list[str]:
        r, hf = self.repo, self.repo / "cache" / "hf"
        return ["docker", "run", "-d", "--rm", "--name", self.name, "--network", "host", "--gpus", "all", "--shm-size", "2g",
                "-u", f"{os.getuid()}:{os.getgid()}", "-w", "/repo",
                "-e", "HOME=/tmp", "-e", "USER=revolve", "-e", "HF_HOME=/hf", "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-e", "PYTHONPATH=/repo/common:/repo/components/evaluation/src", "-v", f"{r}:/repo:ro", "-v", f"{hf}:/hf",
                self.image, "python", "-m", "evaluation.policy_service", "--ckpt", f"/repo/{self.ckpt.relative_to(r)}",
                "--port", str(self.port), "--device", self.device, "--state-dim", str(self.state_dim)]

    def running(self) -> bool:
        r = self.run(["docker", "inspect", "-f", "{{.State.Running}}", self.name], capture_output=True, text=True)
        return r.stdout.strip() == "true"

    def logs(self) -> str:
        r = self.run(["docker", "logs", "--tail", "30", self.name], capture_output=True, text=True)
        return (r.stdout + r.stderr).strip()

    def start(self, timeout_s: float, client=None, sleep=time.sleep) -> None:
        """`docker run` and wait for the model to be loaded and warm. EvalError (with the container's log) otherwise."""
        (self.repo / "cache" / "hf").mkdir(parents=True, exist_ok=True)
        r = self.run(self.argv(), capture_output=True, text=True)
        if r.returncode:
            raise EvalError(f"policy service: docker run failed: {r.stderr.strip()[:500]}")
        client, end = client or PolicyClient(self.url, 5.0), time.monotonic() + timeout_s
        while True:
            try:
                if client.health().get("ready"):
                    return
            except OSError:
                pass
            if not self.running():
                raise EvalError(f"policy service {self.name} exited:\n{self.logs()}")
            if time.monotonic() > end:
                log = self.logs()
                self.stop()
                raise EvalError(f"policy service {self.name}: not ready after {timeout_s:.0f} s:\n{log}")
            sleep(1.0)

    def stop(self) -> None:
        self.run(["docker", "rm", "-f", self.name], capture_output=True)


@REGISTRY.register("policy", "smolvla")
class SmolVLA(Policy):
    name = "smolvla"

    def __init__(self, spec, params):
        if unknown := sorted(set(params) - set(PARAMS)):
            raise EvalError(f"policy smolvla: unknown parameters {unknown} (known: {list(PARAMS)})")
        self.spec, self.params = spec, dict(params)
        self.lock, self.service = threading.Lock(), None

    # ---- seams: what the CPU tests replace
    def ckpt_dir(self) -> Path:
        s = self.spec
        if s.ckpt is None:
            raise EvalError("policy smolvla: needs a `ckpt:`")
        return Path(REGISTRY.create("ckpt", s.ckpt.source, s, s.ckpt.params).resolve(REPO / s.out).path)

    def new_service(self, ckpt: Path, state_dim: int) -> Service:
        return Service(f"{sim_prefix(self.spec)}-policy", ckpt, state_dim, self.params.get("image", IMAGE),
                       self.params.get("device", "cuda"))

    # ----
    def contract(self):
        robot = self.spec.train.robot
        if robot not in CONTRACTS:
            raise EvalError(f"policy smolvla: no observation / action contract for robot {robot!r} (known: {sorted(CONTRACTS)})")
        return CONTRACTS[robot]

    def url(self) -> str:
        """The service's url; it is started by the first caller (one per policy, shared by the run's rollouts)."""
        with self.lock:
            if self.service is None:
                state_dim = self.contract().state_dim
                service = self.new_service(self.ckpt_dir(), state_dim)
                atexit.register(service.stop)
                service.start(float(self.params.get("startup_s", 300)))
                self.service = service
            return self.service.url

    def client_args(self, item):
        per_chunk = int(self.params.get("actions_per_chunk") or self.spec.runner.n_action_steps)
        task = load_task(self.spec.train.task, REPO)
        if task is None:
            raise EvalError(f"policy smolvla: no task {self.spec.train.task!r} under {REPO / 'specs' / 'tasks'}")
        serve = {"url": self.url(), "prompt": task.prompt, "robot": self.spec.train.robot, "actions_per_chunk": per_chunk,
                 "image_size": self.params.get("image_size")}
        return ["--serve", json.dumps(serve)]

    def close(self) -> None:
        with self.lock:
            service, self.service = self.service, None
        if service:
            service.stop()
