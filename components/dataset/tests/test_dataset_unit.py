"""Stub-data unit tests (no docker): labels from episode.json, label selection, the dataset checks, the
progress parser."""
import json

import pytest
import yaml
from revolve_common import specs

from conftest import REPO

import checks as C
import labels as L
import progress as P

CONTRACT = yaml.safe_load((REPO / "specs/robots/mobile_panda/contract.yaml").read_text())
ROBOT = yaml.safe_load((REPO / "specs/robots/mobile_panda/robot.yaml").read_text())
JOINTS = specs.load(REPO / "specs/robots/mobile_panda/robot.yaml").joints   # state / action derived
PROMPT = "open the fridge door"


def episode(tmp_path, name, **block):
    d = tmp_path / name
    d.mkdir()
    (d / "episode.json").write_text(json.dumps({"steps": 13500, "control_dt": 0.001, **block}))
    return d


def test_labels_demo_and_recovery(tmp_path):
    demo = episode(tmp_path, "seed_3_a1", datagen={"seed": 3, "attempt_id": 1, "label": "demo"})
    rec = episode(tmp_path, "ep_0", recovery={"source": "recovery", "phase": "grasp", "suffix": True,
                                              "from_rollout": "/repo/runs/x/turn_1/eval/seed_200", "rollout_seed": 200})
    assert L.of_episode(demo, 0, tmp_path) == dict(source="demo", phase="", suffix=False, seed=3, turn=0,
                                                   from_rollout="", episode_dir="seed_3_a1")
    assert L.of_episode(rec, 2, tmp_path) == dict(source="recovery", phase="grasp", suffix=True, seed=200, turn=2,
                                                  from_rollout="runs/x/turn_1/eval/seed_200", episode_dir="ep_0")
    nos = episode(tmp_path, "ep_1", recovery={"phase": "open", "rollout_seed": None})
    assert L.of_episode(nos, 1)["seed"] == -1 and L.of_episode(nos, 1)["suffix"] is True
    with pytest.raises(ValueError, match="neither"):
        L.of_episode(episode(tmp_path, "odd"), 0)
    cols = L.columns(L.of_episode(rec, 2, tmp_path))
    assert sorted(cols) == sorted("labels/" + k for k in L.NAMES) and L.of_row(cols) == L.of_episode(rec, 2, tmp_path)


def test_select():
    rows = [dict(episode_index=i, **L.columns(dict(source=s, phase=p, suffix=s == "recovery", seed=i, turn=t,
                                                     from_rollout="", episode_dir=f"e{i}")))
            for i, (s, p, t) in enumerate([("demo", "", 0), ("recovery", "open", 1), ("recovery", "grasp", 1),
                                            ("recovery", "open", 2)])]
    assert L.select(rows, source="demo") == [0]
    assert L.select(rows, source="recovery", phase="open") == [1, 3]
    assert L.select(rows, turn=1) == [1, 2]
    assert L.select(rows) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="unknown"):
        L.select(rows, colour="red")


def test_contract_features_match_robot():
    want = C.contract_features(CONTRACT)
    assert want == {"observation.images.wrist": (480, 480, 3), "observation.images.overhead_left": (480, 480, 3),
                    "observation.images.overhead_right": (480, 480, 3), "observation.state": (12,), "action": (11,)}
    # the contract's selects are the robot spec's joints
    assert [s.removeprefix("position.") for s in CONTRACT["observations"]["observation.state"]["select"]] == JOINTS.state
    assert [s.removeprefix("position.") for s in CONTRACT["actions"]["action"]["select"]] == JOINTS.action
    assert len(ROBOT["cameras"]) == sum(k.startswith("observation.images.") for k in want)


def summary(lengths=(502, 404), tasks=(PROMPT,), fps=30, shapes=None, labels=None):
    feats = {k: {"shape": list(v)} for k, v in (shapes or C.contract_features(CONTRACT)).items()}
    feats.update({k: {"shape": [1]} for k in C.EXTRA_FEATURES})
    return dict(info=dict(codebase_version="v3.0", fps=fps, total_episodes=len(lengths), total_frames=sum(lengths),
                          features=feats), lengths=list(lengths), tasks=list(tasks), labels=labels)


def test_dataset_problems():
    sims = [16.72, 13.5]                         # 501 +- 1 and 405 +- 1 frames
    labs = [{"source": "demo"}, {"source": "recovery"}]
    assert C.dataset_problems(summary(labels=labs), sims, 30, PROMPT, CONTRACT, labs) == []
    assert C.expected_frames(4.1, 30) == 123 and C.frames_within_tolerance(404, 13.5, 30)
    assert C.sim_s({"steps": 16720, "control_dt": 0.001}) == 16.72
    bad = C.dataset_problems(summary(lengths=(502, 400)), sims, 30, PROMPT)
    assert bad == ["frames: episode 1 has 400, want 405 +- 1"]
    assert C.dataset_problems(summary(tasks=("open the drawer",)), sims, 30, PROMPT)[0].startswith("tasks:")
    assert C.dataset_problems(summary(), sims[:1], 30, PROMPT)[0].startswith("episode_count:")
    assert C.dataset_problems(summary(fps=15), sims, 30, PROMPT)[0] == "fps: 15 != 30"
    shapes = dict(C.contract_features(CONTRACT), action=(10,))
    assert C.dataset_problems(summary(shapes=shapes), sims, 30, PROMPT, CONTRACT) == ["feature action: shape (10,) != contract (11,)"]
    shapes = {k: v for k, v in C.contract_features(CONTRACT).items() if k != "observation.images.wrist"}
    assert C.dataset_problems(summary(shapes=shapes), sims, 30, PROMPT, CONTRACT)[0].startswith("feature keys:")
    assert C.dataset_problems(summary(labels=labs[:1] * 2), sims, 30, PROMPT, None, labs)[0].startswith("labels:")
    assert C.dataset_problems(summary(), [None, None], 30, PROMPT) == []      # merged: frames not re-checked


def test_progress_parser():
    assert P.parse('PROGRESS {"episodes_converted": 2, "episodes_total": 5}') == {"episodes_converted": 2, "episodes_total": 5}
    assert P.parse("3 / 5 episodes processed") == {"episodes_converted": 3, "episodes_total": 5}
    assert P.parse("Completed: 4/5 episodes (1 failed) in 96.3s") == {"episodes_converted": 4, "episodes_total": 5,
                                                                      "episodes_failed": 1}
    assert P.parse("  -> 405 frames from 04_4_from_open") is None and P.parse("Svt[warn]: Preset M12") is None


def test_rel_paths():
    assert L.rel("/repo/runs/a/b") == "runs/a/b"
    assert L.rel(f"{REPO}/runs/a", REPO) == "runs/a"
    assert L.rel("/elsewhere/x") == "/elsewhere/x"
