"""One training job, independent of where it runs: plan() -> the lerobot-train flags; finalize() -> the checkpoint.

Retrain modes (experiment yaml training.mode):
  finetune_last           init = the previous turn's checkpoint (base on turn 0), the full (merged) dataset
  from_base               init = training.base, the full dataset
  finetune_recovery_only  init = the previous checkpoint (base when none), only the episodes labelled
                          labels/source == recovery (components/dataset labels.select -> --dataset.episodes)
Weights carry over through policy.pretrained_path (as the earlier pipeline: never --resume; the
optimizer, schedule and step count start fresh).

Output: <out>/ = the final checkpoint's pretrained_model/ files (config.json, model.safetensors, train_config.json,
policy_{pre,post}processor*.json + their safetensors) + train.json (mode, init, episodes, steps, losses, sha256).
lerobot-train writes into <out>.partial/lerobot; only a finished run's checkpoint is moved to <out>.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
POLICY = yaml.safe_load((HERE / "smolvla.yaml").read_text())
MODES = ("finetune_last", "from_base", "finetune_recovery_only")
REPO_ID = "revolve/dataset"          # the dataset component's repo id (local only)
TOKEN_EXIT = 3                       # as the earlier pipeline: a Hugging Face token was found
CKPT_FILES = ("config.json", "model.safetensors")


def dataset_labels():
    """components/dataset/src/dataset/labels.py (pure Python), from the repo mount or beside this file."""
    for d in (HERE.parents[2] / "dataset" / "src" / "dataset", Path("/repo/components/dataset/src/dataset")):
        if (d / "labels.py").is_file():
            sys.path.insert(0, str(d))
            import labels
            return labels
    raise SystemExit("labels.py of the dataset component not found")


def recovery_episodes(dataset) -> list[int]:
    from lerobot.datasets import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(REPO_ID, root=dataset)
    return dataset_labels().select(meta.episodes, source="recovery")


def init_of(mode: str, base: str, prev: str | None) -> str:
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    if mode == "from_base" or not prev:
        return base
    if not all((Path(prev) / f).is_file() for f in CKPT_FILES):
        raise ValueError(f"previous checkpoint {prev} lacks {CKPT_FILES}")
    return str(prev)


def plan(*, dataset, out, mode, base, prev=None, steps, batch_size, seed=1000, log_freq=10, num_workers=4,
         job="train", episodes=None) -> dict:
    """-> {"args": {flag: value}, "init", "episodes", "out", "lerobot_out"}; paths as given (the caller maps
    them for a remote machine). episodes: None = every episode, unless finetune_recovery_only (then
    recovery_episodes(dataset) unless given)."""
    init = init_of(mode, base, prev)
    if mode == "finetune_recovery_only" and episodes is None:
        episodes = recovery_episodes(dataset)
        if not episodes:
            raise ValueError(f"finetune_recovery_only: no episode of {dataset} has labels/source == recovery")
    args = {**POLICY["train_args"], "policy.pretrained_path": init, "batch_size": batch_size, "steps": steps,
            "save_freq": steps, "log_freq": log_freq, "seed": seed, "num_workers": num_workers,
            "dataset.repo_id": REPO_ID, "dataset.root": str(dataset),
            "output_dir": f"{out}.partial/lerobot", "job_name": job}
    if episodes is not None:
        args["dataset.episodes"] = "[" + ",".join(map(str, episodes)) + "]"
    return dict(args=args, init=init, mode=mode, episodes=episodes, out=str(out), lerobot_out=args["output_dir"])


def flags(args: dict) -> list[str]:
    """The earlier pipeline's train flags()."""
    return [f"--{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in args.items()]


def argv(p: dict, args: dict | None = None) -> list[str]:
    return [POLICY["entrypoint"], *flags(args or p["args"])]


def refuse_token() -> None:
    """No run may hold a Hugging Face token (legacy train_entry.py): nothing here can push."""
    import huggingface_hub
    if huggingface_hub.get_token():
        print("train: refused, huggingface_hub finds a token (no run may hold one)", flush=True)
        raise SystemExit(TOKEN_EXIT)


def stream(cmd, env=None, on_line=None) -> int:
    """cmd in its own process group with its output teed to ours (the wrapper parses it) and to on_line; the
    group is killed after it exits (datagen entry.call: children must not hold the wrapper's pipe), and a
    SIGTERM (the wrapper's cancel) is passed on, then raised here as SystemExit(143) for the caller's cleanup."""
    import signal
    import subprocess
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         start_new_session=True)

    def kill(sig):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            pass

    def term(*_):
        kill(signal.SIGTERM)
        raise SystemExit(143)

    old = signal.signal(signal.SIGTERM, term)
    try:
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if on_line:
                on_line(line.rstrip("\n"))
        return p.wait()
    finally:
        signal.signal(signal.SIGTERM, old)
        kill(signal.SIGKILL)


def progress(**kw) -> None:
    print("PROGRESS " + json.dumps(kw, default=str), flush=True)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def final_checkpoint(lerobot_out) -> Path:
    """The newest step dir's pretrained_model/ (LeRobot: checkpoints/<step>/, `last` -> it)."""
    steps = sorted(p for p in (Path(lerobot_out) / "checkpoints").glob("[0-9]*") if p.is_dir())
    if not steps:
        raise FileNotFoundError(f"no checkpoint under {lerobot_out}/checkpoints")
    return steps[-1] / "pretrained_model"


def finalize(p: dict, progress: dict | None = None, extra: dict | None = None, ckpt: Path | None = None) -> dict:
    """Moves the final pretrained_model/ to <out>/ (+ train.json) and removes <out>.partial."""
    out, partial = Path(p["out"]), Path(p["out"] + ".partial")
    src = ckpt or final_checkpoint(p["lerobot_out"])
    missing = [f for f in CKPT_FILES if not (src / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{src} lacks {missing}")
    tmp = partial / "ckpt"
    if src != tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.move(str(src), tmp)
    info = dict(mode=p["mode"], init=p["init"], episodes=p["episodes"], steps=p["args"]["steps"],
                batch_size=p["args"]["batch_size"], seed=p["args"]["seed"], dataset=p["args"]["dataset.root"],
                step=Path(src).parent.name, model_sha256=sha256(tmp / "model.safetensors"),
                losses=(progress or {}).get("losses", []), **(extra or {}))
    (tmp / "train.json").write_text(json.dumps(info, indent=1))
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    shutil.rmtree(partial, ignore_errors=True)
    return info
