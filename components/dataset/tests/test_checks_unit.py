"""dataset.checks (DATASET-v2.md sections 8.3, 9; the frame-check decision `abs(frames - floor(duration_s x 30)) <= 6`
plus the exact ticks check): expected_frames and frame_offset, the per-episode checks of convert (convert_episode:
tolerance, exact ticks, late action streams under both values of ACTION_LATE_FAILS) and of merge (merge_episode), and
every problem kind of dataset_problems on a synthetic summary."""
import copy

import pytest

from conftest import PREFIXES
from dataset import checks as K
from dataset.ticks import Ticks

PROMPT = "open the fridge door"
IMAGES = ("observation.images.wrist", "observation.images.overhead_left", "observation.images.overhead_right")
FEATURES = {**{k: (480, 480, 3) for k in IMAGES}, "observation.state": (11,), "action": (11,)}
BOOKKEEPING = ("timestamp", "frame_index", "episode_index", "index", "task_index", "is_first", "is_last",
               "is_terminal")
LENGTHS = [600, 300]
NAME = "runs/fridge/manifest.json entry 4"
HAND = "/panda_hand_controller/joint_trajectory"
ARM = "/panda_arm_controller/joint_trajectory"
CMD_VEL = "/omni_wheel_drive_controller/cmd_vel"


def lab(i):
    return {"source": "demo", "seed": 7, "split": "train", "scene": "train/scene_7/scene.yaml", "episode": i,
            "expert": "e", "attempt": i, "turn": 0, "manifest": "m.json"}


def info(lengths=LENGTHS):
    feats = {k: {"dtype": "video", "shape": [480, 480, 3], "names": ["height", "width", "channels"]} for k in IMAGES}
    feats["observation.state"] = {"dtype": "float64", "shape": [11], "names": [f"s{i}" for i in range(11)]}
    feats["action"] = {"dtype": "float64", "shape": [11], "names": [f"a{i}" for i in range(11)]}
    for k in BOOKKEEPING:
        feats[k] = {"dtype": "bool" if k.startswith("is_") else ("float32" if k == "timestamp" else "int64"),
                    "shape": [1], "names": None}
    return {"codebase_version": "v3.0", "robot_type": "mobile_panda", "fps": 30, "total_episodes": len(lengths),
            "total_frames": sum(lengths), "total_tasks": 1, "features": feats}


def summary(lengths=LENGTHS):
    return {"info": info(lengths), "lengths": list(lengths), "tasks": [PROMPT], "task_indices": [0],
            "labels": [lab(i) for i in range(len(lengths))]}


def prefixed(out):
    assert isinstance(out, list)
    for p in out:
        assert isinstance(p, str) and p.startswith(PREFIXES), p         # each starts with its check's name
    return out


def problems(s, **over):
    kw = dict(fps=30, prompt=PROMPT, features=copy.deepcopy(FEATURES), labels=[lab(0), lab(1)])
    kw.update(over)
    return prefixed(K.dataset_problems(s, **kw))


def convert_episode(frames, duration_s=20.004, ticks=None, late=None):
    """convert_episode of NAME at 30 fps; ticks: the bag's frames (default: equal to frames)."""
    t = Ticks(frames if ticks is None else ticks, 0, late or {})
    probs, warns = K.convert_episode(NAME, frames, duration_s, t, 30)
    return prefixed(probs), prefixed(warns)


def with_lengths(s, lengths):
    s["lengths"] = list(lengths)
    s["info"]["total_frames"] = sum(lengths)
    return s


def only(probs, prefix):
    assert probs, f"expected a {prefix} problem"
    assert all(p.startswith(prefix) for p in probs), probs
    return probs


# ------------------------------------------------------------------ expected_frames / frame_offset

@pytest.mark.parametrize("dur, frames", [(4.1, 123), (8.2, 246), (33.3, 999), (20.004, 600), (10.0, 300), (0, 0),
                                         (0.0, 0), (0.0333, 0), (1 / 30 * 0.99, 0), (20, 600), (16.1, 483),
                                         (19.999, 599)])
def test_expected_frames(dur, frames):
    # checks.expected_frames: floor(duration_s x fps) of the exact product (4.1 x 30 = 123, not 122)
    got = K.expected_frames(dur, 30)
    assert got == frames and isinstance(got, int) and not isinstance(got, bool)


@pytest.mark.parametrize("dur", [-0.001, -1, -20.0])
def test_expected_frames_negative(dur):
    # checks.expected_frames: negative duration -> ValueError
    with pytest.raises(ValueError):
        K.expected_frames(dur, 30)


@pytest.mark.parametrize("frames, dur, off", [(600, 20.004, 0), (606, 20.004, 6), (594, 20.004, -6),
                                              (597, 20.004, -3), (123, 4.1, 0), (122, 4.1, -1)])
def test_frame_offset(frames, dur, off):
    # checks.frame_offset = frames - expected_frames
    assert K.frame_offset(frames, dur, 30) == off


# ------------------------------------------------------------------ convert_episode

@pytest.mark.parametrize("delta", [6, -6, 0])
def test_frame_tolerance_ok(delta):
    # decision: abs(frames - floor(duration_s x 30)) <= 6 is fine (boundary included)
    assert convert_episode(600 + delta) == ([], [])


@pytest.mark.parametrize("delta", [7, -7, -600])
def test_frame_tolerance_exceeded(delta):
    # decision: abs(offset) > 6 -> a `frames:` problem naming the episode
    probs, _ = convert_episode(600 + delta)
    (p,) = probs
    assert p.startswith(f"frames: {NAME}:"), p


@pytest.mark.parametrize("ticks", [601, 599])
def test_ticks_mismatch_by_one(ticks):
    # decision: exact check frames == the porter's ticks minus warmup ticks; off by one is a `frames:` problem
    probs, _ = convert_episode(600, ticks=ticks)
    (p,) = probs
    assert p.startswith(f"frames: {NAME}:") and str(ticks) in p, p


def test_tolerance_and_ticks_both_reported():
    probs, _ = convert_episode(700, ticks=600)
    assert len(probs) == 2, probs


@pytest.mark.parametrize("fails", [False, True])
def test_late_action_streams(monkeypatch, fails):
    # one message per action topic with zero-filled frames (topics at 0 say nothing): a warning while
    # ACTION_LATE_FAILS is False, a problem when it is True; the frame checks are unaffected either way
    monkeypatch.setattr(K, "ACTION_LATE_FAILS", fails)
    probs, warns = convert_episode(600, late={CMD_VEL: 0, ARM: 1, HAND: 4})
    msgs, empty = (probs, warns) if fails else (warns, probs)
    assert empty == []
    assert len(msgs) == 2, msgs
    assert any(m.startswith(f"frames: {NAME}: action {ARM} starts 1 frames after the first frame") for m in msgs), msgs
    assert any(m.startswith(f"frames: {NAME}: action {HAND} starts 4 frames after the first frame") for m in msgs), msgs
    assert not any(CMD_VEL in m for m in msgs), msgs


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("late", [{}, {CMD_VEL: 0, ARM: 0, HAND: 0}], ids=["no-topics", "all-zero"])
def test_nothing_late(monkeypatch, fails, late):
    monkeypatch.setattr(K, "ACTION_LATE_FAILS", fails)
    assert convert_episode(600, late=late) == ([], [])


@pytest.mark.parametrize("fails", [False, True])
def test_late_action_beside_a_frame_problem(monkeypatch, fails):
    monkeypatch.setattr(K, "ACTION_LATE_FAILS", fails)
    probs, warns = convert_episode(600, ticks=601, late={HAND: 4})
    assert len(probs) == (2 if fails else 1) and len(warns) == (0 if fails else 1), (probs, warns)


# ------------------------------------------------------------------ merge_episode

def test_merge_episode_frames_equal_the_inputs():
    # DATASET-v2 9: a merged episode keeps its input dataset's frame count exactly
    assert K.merge_episode(NAME, 600, 600) == []
    for frames in (599, 601):
        (p,) = prefixed(K.merge_episode(NAME, frames, 600))
        assert p.startswith(f"frames: {NAME}:"), p


# ------------------------------------------------------------------ dataset_problems: clean

def test_clean_summary_has_no_problems():
    assert problems(summary()) == []


def test_shape_list_vs_tuple_is_not_a_difference():
    # info.json shapes are JSON lists, contract.features tuples: equal values are equal
    assert problems(summary(), features={k: list(v) for k, v in FEATURES.items()}) == []


def test_bookkeeping_keys_may_be_missing_or_present():
    # checks.BOOKKEEPING: ignored by the feature check (either way)
    s = summary()
    for k in ("is_first", "is_last", "is_terminal"):
        del s["info"]["features"][k]
    assert problems(s) == []


def test_features_of_leaves_out_bookkeeping():
    assert K.features_of(info()) == FEATURES
    assert K.features_of({}) == {}


def test_per_episode_frames_are_not_its_business():
    # dataset_problems checks counts and totals; per-episode frame counts are convert_episode's and merge_episode's
    assert problems(with_lengths(summary(), [700, 100])) == []


# ------------------------------------------------------------------ info: version, fps, counts

def test_codebase_version():
    # DATASET-v2 8.3: codebase_version v3.0
    s = summary()
    s["info"]["codebase_version"] = "v2.1"
    assert problems(s)


def test_codebase_version_missing():
    # DATASET-v2 8.3: codebase_version v3.0 (missing is not v3.0)
    s = summary()
    del s["info"]["codebase_version"]
    assert problems(s)


@pytest.mark.parametrize("fps", [15, 29, 31, None])
def test_fps(fps):
    # DATASET-v2 8.3: fps 30 (the given fps)
    s = summary()
    s["info"]["fps"] = fps
    assert problems(s)


def test_fps_is_the_argument():
    # checks.dataset_problems: info fps compared with the fps argument (a 15 fps dataset checked at 15)
    s = summary()
    s["info"]["fps"] = 15
    assert problems(s, fps=15) == []


def test_episode_count_lengths_vs_labels():
    # DATASET-v2 8.3: episode count == passed entries converted (len(lengths) != len(labels))
    s = with_lengths(summary(), [600])
    s["info"]["total_episodes"] = 1
    s["labels"] = [lab(0)]
    probs = problems(s)
    assert any(p.startswith("episodes:") for p in probs), probs


def test_episode_count_total_episodes():
    # checks.dataset_problems: total_episodes != len(labels)
    s = summary()
    s["info"]["total_episodes"] = 3
    probs = problems(s)
    assert any(p.startswith("episodes:") for p in probs), probs


def test_episode_count_more_episodes_than_expected():
    # checks.dataset_problems: Rosetta converted an extra / the expected labels fewer
    s = summary()
    probs = problems(s, labels=[lab(0)])
    assert any(p.startswith("episodes:") for p in probs), probs


def test_total_frames():
    # DATASET-v2 8.3: sum of episode lengths == total_frames
    s = summary()
    s["info"]["total_frames"] = 901
    assert problems(s)


# ------------------------------------------------------------------ task

@pytest.mark.parametrize("tasks", [[], ["open the fridge"], [PROMPT, PROMPT], [PROMPT, "other"], [PROMPT + " "]])
def test_tasks_must_be_the_prompt(tasks):
    # DATASET-v2 8.3: exactly one task in meta/tasks, == task.yaml prompt -> else `task:`
    s = summary()
    s["tasks"] = tasks
    only(problems(s), "task:")


@pytest.mark.parametrize("idx", [[], [1], [0, 1], [1, 0]])
def test_task_indices(idx):
    # checks.dataset_problems: distinct per-frame task_index values != [0] -> `task:`
    s = summary()
    s["task_indices"] = idx
    only(problems(s), "task:")


def test_prompt_argument_is_compared():
    # DATASET-v2 12: never substitute another prompt: the summary's task vs the prompt given
    only(problems(summary(), prompt="open the top drawer of the cabinet"), "task:")


# ------------------------------------------------------------------ features

def test_feature_missing():
    # checks.dataset_problems: non-bookkeeping keys differ (a contract key missing) -> `features:`
    s = summary()
    del s["info"]["features"]["observation.images.wrist"]
    only(problems(s), "features:")


def test_feature_extra():
    # checks.dataset_problems: a non-bookkeeping key the contract does not have -> `features:`
    s = summary()
    s["info"]["features"]["observation.ground_truth"] = {"dtype": "float64", "shape": [3], "names": None}
    only(problems(s), "features:")


@pytest.mark.parametrize("key, shape", [("observation.state", [10]), ("action", [12]),
                                        ("observation.images.overhead_left", [480, 640, 3]),
                                        ("observation.images.wrist", [3, 480, 480])])
def test_feature_shape(key, shape):
    # checks.dataset_problems: a shape differs -> `features:`
    s = summary()
    s["info"]["features"][key]["shape"] = shape
    only(problems(s), "features:")


@pytest.mark.parametrize("dtype", ["image", "uint8", "float32"])
def test_image_dtype_must_be_video(dtype):
    # checks.dataset_problems / DATASET-v2 5.3 DS5: observation.images.* stored as video
    s = summary()
    s["info"]["features"]["observation.images.overhead_right"]["dtype"] = dtype
    only(problems(s), "features:")


def test_contract_feature_missing_from_expectation():
    # checks.dataset_problems: the expected features drive the check both ways
    feats = dict(FEATURES)
    del feats["action"]
    only(problems(summary(), features=feats), "features:")


# ------------------------------------------------------------------ labels

def test_labels_none():
    # checks.dataset_problems: summary labels None (not read back) -> `labels:`
    s = summary()
    s["labels"] = None
    only(problems(s), "labels:")


def test_labels_value_differs():
    # DATASET-v2 8.3: labels read back == expected labels
    s = summary()
    s["labels"][1]["attempt"] = 9
    only(problems(s), "labels:")


def test_labels_type_differs():
    # DATASET-v2 6: fixed types: an int read back as a str is a different label
    s = summary()
    s["labels"][0]["seed"] = "7"
    only(problems(s), "labels:")


def test_labels_order():
    # DATASET-v2 8.3: labels equal "in order"
    s = summary()
    s["labels"] = list(reversed(s["labels"]))
    only(problems(s), "labels:")


# ------------------------------------------------------------------ several at once

def test_several_problems_all_listed():
    # DATASET-v2 8.4: problems lists them all (one string each)
    s = summary()
    s["tasks"] = ["x"]
    s["labels"] = None
    del s["info"]["features"]["action"]
    probs = problems(s)
    for prefix in ("task:", "labels:", "features:"):
        assert any(p.startswith(prefix) for p in probs), (prefix, probs)
