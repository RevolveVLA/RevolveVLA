"""Checks of a converted / merged dataset against its inputs. Pure Python (unit-tested on the host).
expected_frames, frames_within_tolerance and dataset_problems are ported from the earlier pipeline
(expected_frames, frames_within_tolerance, convert_problems): the dataset summary is dataset_info.py's
{"info", "lengths", "tasks", ...}; the minted episodes become their sim durations (episode.json steps x
control_dt). New here: the feature keys / shapes against the Rosetta contract and the labels read back."""
from __future__ import annotations

import math

CODEBASE_VERSION = "v3.0"
EXTRA_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index",
                  "is_first", "is_last", "is_terminal")      # LeRobot's bookkeeping + Rosetta's boundary markers


def expected_frames(sim_s: float, fps: int) -> int:
    """floor(sim_s x fps) of the exact product. sim_s has at most millisecond resolution, so the
    product is an integer or at least fps/1000 below the next one; the 1e-6 absorbs float error
    (4.1 x 30 is 122.99999999999999 in floats)."""
    if sim_s < 0:
        raise ValueError(f"sim_s must be >= 0, got {sim_s!r}")
    return math.floor(sim_s * fps + 1e-6)


def frames_within_tolerance(count: int, sim_s: float, fps: int) -> bool:
    return abs(count - expected_frames(sim_s, fps)) <= 1


def sim_s(episode: dict) -> float:
    """An episode.json's simulated duration (the bag spans steps x control_dt)."""
    return round(int(episode["steps"]) * float(episode["control_dt"]), 6)


def contract_features(contract: dict) -> dict:
    """{key: shape} the Rosetta contract makes: images (h, w, 3) from their resize, vectors (len(select),)."""
    out = {}
    for key, spec in {**contract.get("observations", {}), **contract.get("actions", {})}.items():
        size = next((a["resize"] for a in spec.get("apply", []) if isinstance(a, dict) and "resize" in a), None)
        out[key] = (int(size[0]), int(size[1]), 3) if size else (len(spec["select"]),)
    return out


def dataset_problems(summary: dict, sims: list[float], fps: int, prompt: str, contract: dict | None = None,
                     labels: list[dict] | None = None) -> list[str]:
    """summary: dataset_info.py's output; sims: each episode's sim duration in episode order (None: not
    checked, e.g. a merged dataset); contract: the Rosetta contract (dict); labels: the expected labels
    per episode."""
    info, lengths, tasks = summary["info"], summary["lengths"], summary["tasks"]
    n = len(sims)
    problems = []
    if info["codebase_version"] != CODEBASE_VERSION:
        problems.append(f"codebase_version: {info['codebase_version']} != {CODEBASE_VERSION}")
    if info["fps"] != fps:
        problems.append(f"fps: {info['fps']} != {fps}")
    if len(lengths) != n or info["total_episodes"] != n:
        problems.append(f"episode_count: {len(lengths)} lengths, total_episodes {info['total_episodes']}, expected {n}")
    if sum(lengths) != info["total_frames"]:
        problems.append(f"total_frames: {info['total_frames']} != sum of lengths {sum(lengths)}")
    if tasks != [prompt]:
        problems.append(f"tasks: {tasks} != [{prompt!r}]")
    for i in range(min(len(lengths), n)):
        if sims[i] is not None and not frames_within_tolerance(lengths[i], sims[i], fps):
            problems.append(f"frames: episode {i} has {lengths[i]}, want {expected_frames(sims[i], fps)} +- 1")
    if contract is not None:
        want = contract_features(contract)
        got = {k: tuple(v["shape"]) for k, v in info["features"].items() if k not in EXTRA_FEATURES}
        if set(got) != set(want):
            problems.append(f"feature keys: {sorted(got)} != contract {sorted(want)}")
        problems += [f"feature {k}: shape {got[k]} != contract {want[k]}" for k in sorted(set(got) & set(want))
                     if got[k] != want[k]]
        if contract.get("fps") != fps:
            problems.append(f"contract fps {contract.get('fps')} != {fps}")
    if labels is not None and summary.get("labels") != labels:
        problems.append(f"labels: {summary.get('labels')} != expected {labels}")
    return problems
