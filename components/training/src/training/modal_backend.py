"""Backend modal: lerobot-train in a Modal Function (experiment training.backends.modal: {gpu: L4, timeout_s,
volume}). Ported from the earlier pipeline's modal_train.py, reduced to one attached call:
- image: debian slim + ffmpeg + uv 0.9.15, the hashed lock synced into /opt/venv (legacy build_image; the same
  lock as revolve-training), no repo code, no secret (`secrets=[]`, legacy push guards: push_to_hub false,
  no token in the env, the trainer refuses a token);
- the dataset (and the previous checkpoint) uploaded to the Volume under jobs/<job>/, lerobot-train's output in
  jobs/<job>/out/lerobot; the Function streams the trainer's lines (modal.enable_output), so the same parser
  serves /status here; the final pretrained_model/ is downloaded, then jobs/<job>/ is removed.
Not ported: detached runs with modal.Retries + resume, the JPEG re-encode / validation split / keep-2 pruning
(legacy-only run keys). Credentials: Modal's own (~/.modal.toml mounted read-only by launch.py, never a token in
a file or command). `modal` is imported lazily, so this module imports without it.
"""
from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import train as T

MOUNT = "/vol"
UV_VERSION = "0.9.15"
LOCK = "/opt/requirements.lock"


def has_credentials() -> bool:
    return bool(os.environ.get("MODAL_TOKEN_ID")) or (Path(os.environ.get("HOME", "/tmp")) / ".modal.toml").is_file()


def remote_train(argv: list, env: dict, volume: str) -> int:
    """Runs in the Modal container: the trainer with its lines printed as they come, then a Volume commit."""
    import subprocess
    import sys

    import modal
    p = subprocess.Popen(argv, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = p.wait()
    modal.Volume.from_name(volume).commit()
    return rc


def image(modal):
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("ffmpeg")
            .pip_install(f"uv=={UV_VERSION}")
            .add_local_file(LOCK, LOCK, copy=True)
            .env(dict(UV_PYTHON_INSTALL_DIR="/opt/uv-python", UV_PYTHON_PREFERENCE="only-managed", UV_NO_CONFIG="1",
                      UV_NO_CACHE="1"))
            .run_commands(f"uv venv --python {T.POLICY['python']} /opt/venv",
                          f"uv pip sync --python /opt/venv/bin/python --require-hashes {LOCK}"))


def download(vol, src: str, dst: Path, file_type) -> int:
    n = 0
    for e in vol.listdir(src, recursive=True):
        if e.type == file_type:
            rel = e.path[len(src.rstrip("/")) + 1:] if e.path.startswith(src.rstrip("/") + "/") else Path(e.path).name
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                vol.read_file_into_fileobj(e.path, f)
            n += 1
    return n


def run(p: dict, settings: dict, job: str, on_line=None, modal=None) -> dict:
    modal = modal or importlib.import_module("modal")
    if not has_credentials():
        raise SystemExit("modal: no Modal credentials on this host (~/.modal.toml); backend not run")
    name = settings.get("volume", "revolve-training")
    vol = modal.Volume.from_name(name, create_if_missing=True)
    base = f"jobs/{job}"
    info = dict(volume=name, dir=base)
    t0 = time.time()
    try:
        T.progress(phase="upload", volume=name, dir=base)
        args = dict(p["args"], **{"dataset.root": f"{MOUNT}/{base}/dataset", "output_dir": f"{MOUNT}/{base}/out/lerobot"})
        with vol.batch_upload(force=True) as b:
            b.put_directory(p["args"]["dataset.root"], f"{base}/dataset")
            if Path(p["init"]).is_dir():
                b.put_directory(p["init"], f"{base}/init")
                args["policy.pretrained_path"] = f"{MOUNT}/{base}/init"
        info["upload_s"] = round(time.time() - t0, 1)
        app = modal.App(f"revolve-train-{job}"[:63])
        fn = app.function(image=image(modal), gpu=settings.get("gpu", "L4"), timeout=int(settings.get("timeout_s", 3600)),
                          volumes={MOUNT: vol}, secrets=[], serialized=True)(remote_train)
        T.progress(phase="train", backend="modal", gpu=settings.get("gpu", "L4"))
        t1 = time.time()
        env = dict(HF_HOME=f"{MOUNT}/hf", PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_TELEMETRY="1",
                   PATH="/opt/venv/bin:/usr/local/bin:/usr/bin:/bin")
        argv = ["/opt/venv/bin/" + T.POLICY["entrypoint"], *T.flags(args)]
        with modal.enable_output(), app.run():
            rc = fn.remote(argv, env, name)
        info["train_s"] = round(time.time() - t1, 1)
        if rc:
            return dict(rc=rc, ckpt=None, remote=info)
        vol.reload()
        steps = sorted(Path(e.path).name for e in vol.listdir(f"{base}/out/lerobot/checkpoints")
                       if Path(e.path).name != "last")
        ckpt = Path(p["out"] + ".partial") / "ckpt"
        t1 = time.time()
        info["files"] = download(vol, f"{base}/out/lerobot/checkpoints/{steps[-1]}/pretrained_model", ckpt,
                                 modal.volume.FileEntryType.FILE)
        info["download_s"], info["step_dir"] = round(time.time() - t1, 1), steps[-1]
        return dict(rc=0, ckpt=ckpt, remote=info)
    finally:
        try:
            vol.remove_file(base, recursive=True)
            info["cleanup"] = "removed"
        except Exception as e:                  # nothing uploaded yet
            info["cleanup"] = f"{type(e).__name__}: {e}"
        T.progress(phase="cleanup", remote=info)
