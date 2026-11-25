"""Checks of a built dataset against its inputs, before it is moved to <out> (DATASET-v2.md section 8.3). Pure Python
(unit-tested on the host): the dataset side is summary.py's {"info", "lengths", "tasks", "task_indices", "labels"}.

  dataset_problems(summary, ...)        both commands: format, counts, task, features, labels
  convert_episode(name, frames, ...)    per converted episode: frames against the evaluator and the bag, late actions
  merge_episode(name, frames, n)        per merged episode: frames against its input dataset

A converted episode's frames, two checks:
1. against the evaluator: abs(frames - floor(duration_s x fps)) <= FRAME_TOLERANCE;
2. exact, against the bag: frames == the porter's ticks over the bag's span minus its warmup ticks (ticks.py)."""
from __future__ import annotations

import math

CODEBASE_VERSION = "v3.0"
# Frames vs floor(duration_s x fps), at most 6 frames (0.2 s at 30 fps) apart. Measured on simulator bags
# (DATASET-v2.md section 8.3): a bag starts 16-83 ms after its episode and ends 38-109 ms after it, and the porter
# skips 1-3 warmup ticks until every observation stream has a sample, so six real bags came out at -3 .. +2 frames;
# DS6's +-1 does not hold. The bound is the document's proposal, pending the project owner's confirmation.
FRAME_TOLERANCE = 6
# A late action stream (its first message after the first emitted frame) leaves zeros in that action until it arrives:
# Rosetta gates its warmup on the observation streams only. Recorded per episode and warned about; whether it fails the
# episode is pending the project owner's decision. This flag is the whole policy.
ACTION_LATE_FAILS = False
BOOKKEEPING = ("timestamp", "frame_index", "episode_index", "index", "task_index",   # LeRobot's
               "is_first", "is_last", "is_terminal")                                # Rosetta's episode markers


def expected_frames(duration_s: float, fps: int) -> int:
    """floor(duration_s x fps) of the exact product: durations have at most microsecond resolution, so the 1e-6
    only absorbs float error (4.1 x 30 is 122.99999999999999 in floats)."""
    if duration_s < 0:
        raise ValueError(f"duration_s must be >= 0, got {duration_s!r}")
    return math.floor(duration_s * fps + 1e-6)


def frame_offset(frames: int, duration_s: float, fps: int) -> int:
    return frames - expected_frames(duration_s, fps)


def features_of(info: dict) -> dict:
    """{key: shape} of meta/info.json's features, LeRobot's and Rosetta's bookkeeping columns left out."""
    return {k: tuple(v["shape"]) for k, v in info.get("features", {}).items() if k not in BOOKKEEPING}


def convert_episode(name: str, frames: int, duration_s: float, ticks, fps: int) -> tuple[list[str], list[str]]:
    """One converted episode's frames -> (problems, warnings). ticks: its bag's ticks.Ticks. Problems: frames off
    floor(duration_s x fps) by more than FRAME_TOLERANCE, or not the bag's ticks minus warmup. A late action stream
    (frames zero-filled at the start) is a warning, a problem when ACTION_LATE_FAILS."""
    out = []
    off = frame_offset(frames, duration_s, fps)
    if abs(off) > FRAME_TOLERANCE:
        out.append(f"frames: {name}: {frames} frames, {off:+d} from floor(duration_s {duration_s} x {fps}) = "
                   f"{expected_frames(duration_s, fps)} (tolerance {FRAME_TOLERANCE})")
    if frames != ticks.frames:
        out.append(f"frames: {name}: {frames} frames, the bag's ticks minus warmup are {ticks.frames}")
    late = [f"frames: {name}: action {topic} starts {n} frames after the first frame (zero-filled until then)"
            for topic, n in ticks.action_late.items() if n > 0]
    return (out + late, []) if ACTION_LATE_FAILS else (out, late)


def merge_episode(name: str, frames: int, expected: int) -> list[str]:
    """One merged episode's frames against its input dataset's."""
    return [] if frames == expected else [f"frames: {name}: {frames} frames, its input dataset has {expected}"]


def dataset_problems(summary: dict, *, fps: int, prompt: str, features: dict, labels: list[dict]) -> list[str]:
    """A built dataset (summary.py's output) against what it must hold, for convert and merge alike: format, fps,
    episode and frame counts, the one task, the contract's features (contract.features()), the expected labels per
    episode. The per-episode frame checks are convert_episode and merge_episode."""
    info, lengths = summary["info"], summary["lengths"]
    n = len(labels)
    out = []
    if info.get("codebase_version") != CODEBASE_VERSION:
        out.append(f"features: codebase_version {info.get('codebase_version')} != {CODEBASE_VERSION}")
    if info.get("fps") != fps:
        out.append(f"features: fps {info.get('fps')} != {fps}")
    if len(lengths) != n or info.get("total_episodes") != n:
        out.append(f"episodes: {len(lengths)} in meta/episodes, total_episodes {info.get('total_episodes')}, "
                   f"expected {n}")
    if sum(lengths) != info.get("total_frames"):
        out.append(f"frames: total_frames {info.get('total_frames')} != the episodes' sum {sum(lengths)}")
    if summary["tasks"] != [prompt]:
        out.append(f"task: meta/tasks holds {summary['tasks']}, expected exactly [{prompt!r}]")
    if summary.get("task_indices") != [0]:
        out.append(f"task: the frames carry task indices {summary.get('task_indices')}, expected only 0 (the prompt)")
    got = features_of(info)
    if set(got) != set(features):
        out.append(f"features: keys {sorted(got)} != the contract's {sorted(features)}")
    out += [f"features: {k} shape {got[k]} != the contract's {features[k]}" for k in sorted(set(got) & set(features))
            if got[k] != tuple(features[k])]
    out += [f"features: {k} dtype {info['features'][k].get('dtype')} != video" for k in sorted(got)
            if k.startswith("observation.images.") and info["features"][k].get("dtype") != "video"]
    if summary.get("labels") != labels:
        got_labels = summary.get("labels") or []
        bad = [i for i in range(max(n, len(got_labels)))
               if i >= n or i >= len(got_labels) or got_labels[i] != labels[i]]
        out.append(f"labels: meta/episodes labels differ from the expected ones at episodes {bad[:10]}"
                   + ("" if summary.get("labels") is not None else " (no labels/* columns)"))
    return out
