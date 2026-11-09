"""Host side: run training and serve jobs in revolve-training (docker SDK) under revolve_common.wrapper.

  res = train(dataset, out, mode=..., base=..., prev=None, steps=..., batch_size=..., backend="local",
              settings={})                       # -> out/ = the final checkpoint + train.json
  res = train_from_spec(training, dataset, out, prev)   # the experiment's training block (specs.Training)
  job = start_train(...); res = finish(job)      # start = non-blocking; /status on job.ip():8000
  job = start_serve(ckpt, job_dir, network)      # policy server on an INTERNAL docker network only
  code, out = in_image(argv, gpu=False, network=None)   # one-off command (tests, parity)

Mounts: repo read-only at /repo, runs/ read-write, cache/hf read-write at /hf (HF_HOME: lerobot/smolvla_base and
SmolVLM2, public, no token). Backends (the experiment's training.backend):
  local        the container gets the GPU and runs lerobot-train itself
  gpu_machine  a CPU container drives ssh/rsync to the machine (components/training/src/training/gpu_machine.py);
               the user's ~/.ssh and /etc/passwd + /etc/group are mounted read-only (ssh needs the uid's entry).
               The host (user@host, passwordless ssh) is $REVOLVE_GPU_MACHINE_HOST unless the settings give one.
  modal        a CPU container runs the Modal client; ~/.modal.toml is mounted read-only when it exists
Jobs use docker's bridge network for the wrapper's HTTP (no published ports). Serve refuses a network that is
not --internal: the server unpickles what clients send.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pwd
import sys
import time
import uuid
from pathlib import Path

import docker

from revolve_common import containers
from revolve_common import gpu as gpu_health
from revolve_common import job as J

REPO = Path(__file__).resolve().parents[2]
IMAGE = os.environ.get("REVOLVE_TRAINING_IMAGE", "revolve-training:latest")
SRC = "/repo/components/training/src/training"
ENTRY = f"{SRC}/entry.py"
HF = REPO / "cache" / "hf"
ROSETTA = "/repo/ThirdParty/rosetta/policies/lerobot/lerobot_rosetta"
GPU_HOST_ENV = "REVOLVE_GPU_MACHINE_HOST"
_s = importlib.util.spec_from_file_location("sim_launch", REPO / "components" / "simulator" / "launch.py")
sim_launch = sys.modules.setdefault("sim_launch", importlib.util.module_from_spec(_s))
if not hasattr(sim_launch, "finish"):
    _s.loader.exec_module(sim_launch)
cpath, finish, Job, LABEL = sim_launch.cpath, sim_launch.finish, sim_launch.Job, sim_launch.LABEL


def _arg(a) -> str:
    return cpath(a) if str(a).startswith(str(REPO) + "/") else str(a)


def _base_kw(env=None, gpu=False):
    HF.mkdir(parents=True, exist_ok=True)
    # USER: getpass.getuser() (torch inductor) needs it when the uid has no passwd entry
    kw = dict(user=f"{os.getuid()}:{os.getgid()}", working_dir="/repo", shm_size="8g",
              environment={"HOME": "/tmp", "USER": "revolve", "HF_HOME": "/hf", "PYTHONDONTWRITEBYTECODE": "1", "REPO": "/repo",
                           "PYTHONPATH": f"/repo/common:{SRC}:{ROSETTA}", **(env or {})},
              volumes={str(REPO): {"bind": "/repo", "mode": "ro"}, str(REPO / "runs"): {"bind": "/repo/runs", "mode": "rw"},
                       str(HF): {"bind": "/hf", "mode": "rw"}})
    if gpu:
        gpu_health.launch_check()             # a GPU in a broken state: fail now, not at the job's timeout
        kw["device_requests"] = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
    return kw


def _backend_kw(backend, kw):
    if backend == "gpu_machine":
        home = pwd.getpwuid(os.getuid()).pw_dir
        kw["environment"]["HOME"] = home
        kw["volumes"].update({"/etc/passwd": {"bind": "/etc/passwd", "mode": "ro"},
                              "/etc/group": {"bind": "/etc/group", "mode": "ro"},
                              f"{home}/.ssh": {"bind": f"{home}/.ssh", "mode": "ro"}})
    elif backend == "modal":
        toml = Path.home() / ".modal.toml"
        if toml.is_file():
            kw["volumes"][str(toml)] = {"bind": "/tmp/.modal.toml", "mode": "ro"}
    return kw


def gpu_machine_settings(settings) -> dict:
    """gpu_machine settings with `host` from $REVOLVE_GPU_MACHINE_HOST when the experiment gives none."""
    s = dict(settings or {})
    s["host"] = s.get("host") or os.environ.get(GPU_HOST_ENV, "")
    if not s["host"]:
        raise RuntimeError(f"training backend gpu_machine: no GPU machine host. Set {GPU_HOST_ENV}=user@host "
                           "(passwordless ssh) or training.backends.gpu_machine.host in the experiment yaml")
    return s


def _job(job_dir, args, component_args, parser, timeout, linger_s, kw, name=None) -> Job:
    job_dir = Path(job_dir) if Path(job_dir).is_absolute() else REPO / job_dir
    job_dir.mkdir(parents=True, exist_ok=True)
    for f in (J.STATUS, J.RESULT, J.LOG):
        (job_dir / f).unlink(missing_ok=True)
    cmd = ["python3", ENTRY, *map(_arg, args)]
    jid = f"training_{job_dir.name}_{uuid.uuid4().hex[:6]}"
    J.write_model(job_dir / J.JOB, J.Job(id=jid, component="training", command=cmd, params=component_args))
    c = containers.run(
        docker.from_env(), IMAGE, ["python3", "-m", "revolve_common.wrapper", "--job-dir", cpath(job_dir),
                                   "--parser", parser, "--linger-s", str(linger_s)],
        detach=True, labels={LABEL: "training", "revolve.job": jid}, name=name, **kw)
    return Job(c, job_dir, timeout, time.time())


def start_train(dataset, out, *, mode="finetune_last", base="lerobot/smolvla_base", prev=None, steps, batch_size,
                backend="local", settings=None, seed=1000, log_freq=10, num_workers=4, job_dir=None,
                timeout=4 * 3600.0, linger_s=0.0, network=None) -> Job:
    out = Path(out)
    job_dir = job_dir or out.parent / "jobs" / f"train_{out.name}"
    name = f"{Path(job_dir).parent.parent.name}_{Path(job_dir).name}"
    if backend == "gpu_machine":
        settings = gpu_machine_settings(settings)
    args = ["train", "--backend", backend, "--dataset", dataset, "--out", out, "--mode", mode, "--base", base,
            "--steps", steps, "--batch-size", batch_size, "--seed", seed, "--log-freq", log_freq,
            "--num-workers", num_workers, "--settings", json.dumps(settings or {}), "--job", name]
    if prev:
        args += ["--prev", prev]
    kw = _backend_kw(backend, _base_kw(gpu=backend == "local"))
    kw["network"] = network
    return _job(job_dir, args, dict(args=list(map(str, args))), "progress:parse", timeout, linger_s, kw)


def train(dataset, out, **kw) -> dict:
    return finish(start_train(dataset, out, **kw))


def train_from_spec(training, dataset, out, prev=None, **kw) -> dict:
    """training: the experiment's specs.Training block (mode, base, steps, batch_size, backend, backends)."""
    return train(dataset, out, mode=training.mode, base=training.base, prev=prev, steps=training.steps,
                 batch_size=training.batch_size, backend=training.backend,
                 settings=training.backends.get(training.backend, {}), **kw)


def start_serve(ckpt, job_dir, network, name=None, device="cuda", port=8080, timeout=24 * 3600.0) -> Job:
    """Rosetta's policy server on `network`, which must be an internal docker network; clients reach it as
    <name>:<port>. The wrapper (HTTP :8000) reports ready + chunks; cancel or finish() stops it."""
    net = docker.from_env().networks.get(network)
    if not net.attrs.get("Internal"):
        raise ValueError(f"serve: network {network} is not internal (docker network create --internal); the server "
                         "unpickles client data")
    kw = _base_kw({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, gpu=device.startswith("cuda"))
    kw.update(network=network, working_dir="/tmp")      # LeRobot's server writes logs/ into its cwd
    args = ["serve", "--ckpt", ckpt, "--port", port, "--device", device]
    return _job(job_dir, args, dict(args=list(map(str, args))), "progress:parse_serve", timeout, 0.0, kw,
                name=name or f"serve_{uuid.uuid4().hex[:8]}")


def stop(job: Job, timeout=20) -> dict:
    """Stops a serve job (SIGTERM -> the wrapper cancels the server) and collects it like finish()."""
    try:
        job.container.stop(timeout=timeout)
    except docker.errors.NotFound:
        pass
    job.timeout = time.time() - job.t0 + 30
    return finish(job)


def in_image(argv, gpu=False, network=None, timeout=900, env=None) -> tuple[int, str]:
    kw = _base_kw({"HF_HUB_OFFLINE": "1", **(env or {})}, gpu=gpu)
    c = containers.run(docker.from_env(), IMAGE, list(map(_arg, argv)), detach=True, labels={LABEL: "training"},
                       network=network, **({} if network else {"network_mode": "none"}), **kw)
    try:
        code = c.wait(timeout=timeout)["StatusCode"]
        return code, c.logs().decode()
    finally:
        c.remove(force=True)
