"""Backend gpu_machine: lerobot-train on a throwaway ssh GPU machine (experiment training.backends.gpu_machine:
{host: user@ip, scratch: ~/revolve_scratch, gpu: 0}).

Per job everything lives in <scratch>/<job>/ and is deleted afterwards, whatever happened:
  bin/uv + bin/requirements.lock      this image's uv binary and the hashed lock (scp'd)
  venv/, cache/{uv,python,hf,...}     the env, built there from the lock (legacy Machine.ensure_venv), every
                                      cache and HOME pointed inside, so nothing is written outside the job dir
  data/dataset/, init/                the dataset and the previous checkpoint (rsync)
  out/lerobot/                        lerobot-train's output; the final pretrained_model/ is rsync'd back
Why a venv and not the image (`docker save | docker load`): the env is ~5.5 GB built from PyPI at the
machine's ~55 MB/s vs ~6 GB of image pushed over the ~17 MB/s link, and the machine's disk had 14 GB free (a
loaded image plus its tarball would not fit); the venv also needs no docker state outside the job dir.
lerobot-train runs as `ssh host <script>` under the job's stream(): its output comes back on our stdout, so
the wrapper's parser serves /status here. Cleanup: kill every process whose command line holds the job dir,
rm -rf it, rmdir the scratch dir when empty; then the job dir must be gone.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import train as T

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]
UV = "/usr/local/bin/uv"
LOCK = "/opt/requirements.lock"
NAME = re.compile(r"[^A-Za-z0-9._-]")


class Remote:
    def __init__(self, host, runner=subprocess.run):
        self.host, self.runner = host, runner

    def sh(self, script, timeout=600, check=True) -> str:
        r = self.runner([*SSH, self.host, script], capture_output=True, text=True, timeout=timeout)
        if check and r.returncode:
            raise RuntimeError(f"ssh {self.host}: exit {r.returncode}: {(r.stderr or r.stdout)[-2000:]}")
        return r.stdout

    def rsync(self, src, dst, timeout=7200) -> float:
        t0 = time.time()
        r = self.runner(["rsync", "-a", "-L", "--partial", "-e", " ".join(SSH), src, dst], capture_output=True,
                        text=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f"rsync {src} -> {dst}: exit {r.returncode}: {r.stderr[-2000:]}")
        return round(time.time() - t0, 1)


def env_script(s: str, gpu) -> str:
    """Exports that keep every write of the job inside s (its HOME, the uv/HF/XDG/torch caches)."""
    c = f"{s}/cache"
    env = dict(HOME=f"{s}/home", UV_CACHE_DIR=f"{c}/uv", UV_PYTHON_INSTALL_DIR=f"{c}/python", UV_NO_CONFIG="1",
               UV_PYTHON_PREFERENCE="only-managed", XDG_CACHE_HOME=f"{c}/xdg", HF_HOME=f"{c}/hf",
               TRITON_CACHE_DIR=f"{c}/triton", TORCHINDUCTOR_CACHE_DIR=f"{c}/inductor", CUDA_CACHE_PATH=f"{c}/nv",
               MPLCONFIGDIR=f"{c}/mpl", PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_TELEMETRY="1",
               CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=f"{s}/tmp")
    return ("set -euo pipefail; unset HF_TOKEN HUGGING_FACE_HUB_TOKEN; "
            + " ".join(f"export {k}={shlex.quote(v)};" for k, v in env.items())
            + f' mkdir -p "$HOME" "$TMPDIR"; cd {shlex.quote(s)}; ')


def cleanup_script(s: str, scratch: str) -> str:
    # "[/]home/..." matches the job's processes but not this shell's own command line
    pat = shlex.quote(f"[{s[0]}]{s[1:]}/")
    q = shlex.quote
    return (f"pkill -TERM -f {pat}; sleep 2; pkill -KILL -f {pat}; rm -rf -- {q(s)}; "
            f"rmdir {q(scratch)} 2>/dev/null; test -e {q(s)} && echo LEFT || echo CLEAN")


def run(p: dict, settings: dict, job: str, on_line=None, stream=T.stream, runner=subprocess.run) -> dict:
    """-> {"rc", "ckpt": local dir of the final pretrained_model (in <out>.partial), "remote": timings}."""
    r = Remote(settings["host"], runner)
    home = r.sh("echo $HOME").strip()
    scratch = settings.get("scratch", "~/revolve_scratch").replace("~", home, 1).rstrip("/")
    s = f"{scratch}/{NAME.sub('_', job)}"
    info = dict(host=settings["host"], dir=s)
    t0 = time.time()
    try:
        r.sh(f"mkdir -p {shlex.quote(s)}/bin {shlex.quote(s)}/data {shlex.quote(s)}/out")
        T.progress(phase="upload", remote_dir=s)
        info["upload_tools_s"] = r.rsync(f"{UV}", f"{settings['host']}:{s}/bin/uv") + \
            r.rsync(LOCK, f"{settings['host']}:{s}/bin/requirements.lock")
        info["upload_dataset_s"] = r.rsync(f"{p['args']['dataset.root']}/", f"{settings['host']}:{s}/data/dataset/")
        args = dict(p["args"], **{"dataset.root": f"{s}/data/dataset", "output_dir": f"{s}/out/lerobot"})
        if Path(p["init"]).is_dir():
            info["upload_init_s"] = r.rsync(f"{p['init']}/", f"{settings['host']}:{s}/init/")
            args["policy.pretrained_path"] = f"{s}/init"
        env = env_script(s, settings.get("gpu", 0))
        T.progress(phase="env")
        t1 = time.time()
        rc = stream([*SSH, settings["host"], env
                     + f"bin/uv venv -q --python {T.POLICY['python']} venv && "
                     "bin/uv pip sync -q --python venv/bin/python --require-hashes bin/requirements.lock && "
                     "du -sh venv cache | tr '\\t' ' '"], dict(os.environ), on_line)
        info["env_s"] = round(time.time() - t1, 1)
        if rc:
            return dict(rc=rc, ckpt=None, remote=info)
        T.progress(phase="train", backend="gpu_machine", env_s=info["env_s"])
        t1 = time.time()
        token_check = ("venv/bin/python -c 'import huggingface_hub, sys; "
                       "sys.exit(3 if huggingface_hub.get_token() else 0)' || "
                       "{ echo 'train: refused, huggingface_hub finds a token'; exit 3; }; ")
        rc = stream([*SSH, settings["host"], env + token_check + "exec nice -n 10 venv/bin/"
                     + " ".join(shlex.quote(a) for a in T.argv(p, args))], dict(os.environ), on_line)
        info["train_s"] = round(time.time() - t1, 1)
        if rc:
            return dict(rc=rc, ckpt=None, remote=info)
        step = r.sh(f"ls {shlex.quote(s)}/out/lerobot/checkpoints | grep -v '^last$' | sort | tail -1").strip()
        ckpt = Path(p["out"] + ".partial") / "ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        T.progress(phase="download", step=step)
        info["download_s"] = r.rsync(f"{settings['host']}:{s}/out/lerobot/checkpoints/{step}/pretrained_model/",
                                     f"{ckpt}/")
        info["step_dir"] = step
        return dict(rc=0, ckpt=ckpt, remote=info)
    finally:
        info["cleanup"] = r.sh(cleanup_script(s, scratch), timeout=120, check=False).strip().splitlines()[-1:]
        info["total_s"] = round(time.time() - t0, 1)
        T.progress(phase="cleanup", remote=info)
        if info["cleanup"] != ["CLEAN"]:
            raise RuntimeError(f"gpu_machine: {s} left on {settings['host']}: {info['cleanup']}")
