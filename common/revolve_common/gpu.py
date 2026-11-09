"""GPU health preflight: fail fast instead of hanging on a GPU in a broken state.

A GPU after an Xid 119/154 ("GPU requires reset") still makes `nvidia-smi` exit 0 and the CSV queries look normal,
but its table shows `ERR!` (and `nvidia-smi -q` "GPU requires reset"); running sims then hang until their
watchdogs and new GPU containers fail with "nvml error: gpu requires reset".

  gpu.check()                       host nvidia-smi; "ok", or "not checked" without nvidia-smi; raises GpuError
  gpu.check_docker(dk, image)       nvidia-smi in a throwaway container of `image` (the GPU with utility caps only:
                                    no workload); a container docker cannot start is a GpuError too
  gpu.launch_check()                for the component launchers: check() at most every CACHE_S while healthy

REVOLVE_GPU_PREFLIGHT=0 turns every check off (returns "disabled").
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

BAD = ("ERR!", "requires reset", "NVIDIA-SMI has failed", "Unable to determine the device handle",
       "No devices were found")
CACHE_S = 60.0
_last_ok = 0.0


class GpuError(RuntimeError):
    pass


def enabled() -> bool:
    return os.environ.get("REVOLVE_GPU_PREFLIGHT", "1") != "0"


def problem(returncode, output) -> str | None:
    """None when nvidia-smi's exit code and output look healthy, else what is wrong."""
    if returncode != 0:
        lines = [x for x in output.strip().splitlines() if x.strip()]
        return f"nvidia-smi exit {returncode}" + (f": {lines[-1].strip()}" if lines else "")
    hits = [b for b in BAD if b in output]
    return f"nvidia-smi reports {', '.join(repr(h) for h in hits)}" if hits else None


def _fail(what):
    raise GpuError(f"GPU preflight failed: {what}. The GPU needs a reset (reboot); `nvidia-smi` shows its state. "
                   "Set REVOLVE_GPU_PREFLIGHT=0 to skip this check.")


def check(timeout_s=30.0, smi="nvidia-smi") -> str:
    if not enabled():
        return "disabled"
    exe = shutil.which(smi)
    if exe is None:
        return "not checked"
    try:
        p = subprocess.run([exe], capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _fail(f"nvidia-smi did not answer within {timeout_s:.0f} s")
    err = problem(p.returncode, p.stdout + p.stderr)
    if err:
        _fail(err)
    return "ok"


def check_docker(dk, image, timeout_s=60.0) -> str:
    if not enabled():
        return "disabled"
    import docker
    from docker.types import DeviceRequest
    c = None
    try:
        c = dk.containers.create(image, [], entrypoint=["nvidia-smi"], network_mode="none",
                                 labels={"revolve.role": "gpu_preflight"},
                                 environment={"NVIDIA_VISIBLE_DEVICES": "all", "NVIDIA_DRIVER_CAPABILITIES": "utility"},
                                 device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])])
        try:
            c.start()
        except docker.errors.APIError as e:
            _fail(f"docker could not start a GPU container: {e.explanation or e}")
        try:
            code = c.wait(timeout=timeout_s)["StatusCode"]
        except Exception:
            _fail(f"nvidia-smi did not answer within {timeout_s:.0f} s")
        err = problem(code, c.logs().decode(errors="replace"))
        if err:
            _fail(err)
        return "ok"
    finally:
        if c is not None:
            try:
                c.remove(force=True)
            except docker.errors.APIError:
                pass


def launch_check() -> str:
    global _last_ok
    if time.time() - _last_ok < CACHE_S:
        return "ok"
    r = check()
    if r == "ok":
        _last_ok = time.time()
    return r
