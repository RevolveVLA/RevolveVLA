"""judges c0/c1/c2 (plugins/judges) on synthetic state logs, the judge default / spec error, the pipeline's call."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from datagen.core.spec import SpecError

from evaluation.core import spec as S
from evaluation.core.errors import EvalError
from evaluation.core.pipeline import Pipeline, PluginSet
from evaluation.core.registry import REGISTRY
from evaluation.core.task import load_task

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[2]
HINGE = np.array([-1.2, 2.4, 1.75])
GRASP = [[0, 0, 0, -1.79], [0, 0, 0, 2.35], [0, 0, 0, 0.97], [0, 0, 0, 1.0]]
DT, N = 0.001, 14000
TRAIN = {
    "task": "open_fridge_door", "robot": "r", "out": "train_run", "generator": {"plugin": "fake_datagen"},
    "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
    "splits": {"train": {"target": 3, "pool": {"initial": [0, 9]}}},
}


@pytest.fixture
def task():
    return load_task("open_fridge_door", REPO)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(TRAIN))
    shutil.copytree(REPO / "specs" / "tasks" / "open_fridge_door", tmp_path / "specs" / "tasks" / "open_fridge_door")
    return tmp_path


def judge(name, *params):
    return REGISTRY.create("judge", name, None, dict(params))


def rollout_dir(tmp_path, open_deg=65.0, held=True):
    """A rollout dir whose door opens 0 -> open_deg in 10 s and stays; the TCP follows the bar (held) or stays away (push)."""
    t = np.arange(N) * DT
    q = np.radians(open_deg) * np.clip((t - 1.0) / 10.0, 0, 1)
    d = np.array(GRASP)[:2, 3] - HINGE[:2]
    c, s = np.cos(q), np.sin(q)
    bar = np.stack([HINGE[0] + c * d[0] - s * d[1], HINGE[1] + s * d[0] + c * d[1]], 1)
    pos = np.column_stack([bar if held else bar + 0.5, np.full(N, 1.0)])
    tcp = np.column_stack([pos, np.tile([1.0, 0, 0, 0], (N, 1))])
    names = np.array(["base_x", "panda_finger_joint_left", "panda_finger_joint_right"])
    np.savez(tmp_path / "state_log.npz", door_pos=q, action_grip=np.zeros(N), tcp_pose=tcp, handle_in_tcp=HINGE - pos,
             control_dt=np.array(DT), robot_joint_names=names,
             robot_joint_pos=np.column_stack([np.zeros(N), np.full(N, 0.007), np.full(N, 0.007)]),
             contact_left_finger_handle=np.ones(N, bool), contact_right_finger_handle=np.ones(N, bool))
    (tmp_path / "live.json").write_text(json.dumps({"grasp_pose_world": GRASP}))
    return tmp_path


@pytest.mark.parametrize("kw, c0, c1, c2", [
    (dict(), True, True, True),                                  # opened while held
    (dict(held=False), True, True, False),                       # opened by a push
    (dict(open_deg=40.0), False, False, False),                  # not opened
])
def test_verdicts_on_state_logs(tmp_path, task, kw, c0, c1, c2):
    d = rollout_dir(tmp_path, **kw)
    record = {"passed": c0}
    got = [judge(n).score(record, d, task) for n in ("c0", "c1", "c2")]
    assert [v.passed for v in got] == [c0, c1, c2]
    assert [v.detail["criterion"] for v in got] == ["c0", "c1", "c2"]
    assert got[2].detail["door_max_deg"] == (65.0 if kw.get("open_deg", 65.0) == 65.0 else 40.0)


def test_a_retry_gets_the_phase_verdict_from_the_restore(tmp_path, task):
    d = rollout_dir(tmp_path, held=False)                        # the log would fail C2
    record = {"passed": True, "restore": {"source": "x", "phase": "open"}}
    for name in ("c1", "c2"):
        v = judge(name).score(record, d, task)
        assert v.passed and v.detail == {"criterion": "c0_from_restore"}
    assert not judge("c2").score(record | {"passed": False}, d, task).passed


def test_a_missing_log_fails_with_the_reason_and_a_task_without_rule_is_an_error(tmp_path, task):
    v = judge("c2").score({"passed": True}, tmp_path, task)
    assert not v.passed and "FileNotFoundError" in v.detail["error"] and v.detail["criterion"] == "c2"
    with pytest.raises(EvalError, match="judge c1: the task has no `success:` block"):
        judge("c1").score({"passed": True}, tmp_path, None)
    with pytest.raises(EvalError, match="takes no parameters"):
        judge("c0", ("x", 1))


def parse(root, task="open_fridge_door", **edits):
    (root / "train.yaml").write_text(yaml.safe_dump(TRAIN | {"task": task}))
    raw = {"train": "train.yaml", "out": "run", "sets": {"h": {"count": 1, "pool": {"initial": [100, 100]}}}} | edits
    return S.parse(raw, "eval.yaml", root)


def test_default_judge_follows_the_tasks_success_rule(root):
    shutil.copytree(REPO / "specs" / "tasks" / "open_drawer", root / "specs" / "tasks" / "open_drawer")
    assert parse(root).judge.plugin == "c2"
    assert parse(root, "open_drawer").judge.plugin == "c0"
    assert parse(root, "no_such_task").judge.plugin == "c0"
    assert parse(root, judge={"criterion": "c0"}).judge.plugin == "c0"


@pytest.mark.parametrize("criterion", ["c1", "c2"])
def test_c1_c2_on_a_task_without_success_is_a_spec_error(root, criterion):
    shutil.copytree(REPO / "specs" / "tasks" / "open_drawer", root / "specs" / "tasks" / "open_drawer")
    with pytest.raises(SpecError, match=rf"eval.yaml.judge.criterion: {criterion} needs a `success:` block in task open_drawer"):
        parse(root, "open_drawer", judge={"criterion": criterion})
    assert parse(root, "open_drawer", judge={"criterion": "c0"}).judge.plugin == "c0"


def test_the_pipeline_gives_the_judge_the_rollout_dir_and_the_task(root, monkeypatch):
    monkeypatch.syspath_prepend(str(TESTS))
    import fake_eval as F
    seen = []

    class Spy(F.Judge):
        def score(self, record, rollout_dir, task):
            seen.append((rollout_dir, task.name))
            return F.Verdict(True)

    spec = parse(root, sets={"h": {"plugin": "fake_eval", "count": 1, "pool": {"initial": [100, 100]}}})
    p = PluginSet({"h": F.FakeSet(spec)}, None, F.FakePolicy(spec, {}), F.FakeRollout(spec, {}), Spy())
    assert Pipeline(spec, p, root / "run", log=lambda m: None).run(["rollout"])
    assert seen == [(root / "run" / "h" / "100_r0", "open_fridge_door")]


def test_the_success_block_changes_no_datagen_spec_hash(tmp_path):
    """task.yaml is not part of DatagenSpec.hash: dropping the block from a copy leaves the hash of the training spec."""
    from datagen.core.spec import load
    shutil.copytree(REPO / "specs", tmp_path / "specs", ignore=shutil.ignore_patterns("matrix"))
    path = tmp_path / "specs" / "datagen" / "fridge_base50.yaml"
    before = load(path).hash
    task_yaml = tmp_path / "specs" / "tasks" / "open_fridge_door" / "task.yaml"
    d = yaml.safe_load(task_yaml.read_text())
    d.pop("success")
    task_yaml.write_text(yaml.safe_dump(d))
    assert load(path).hash == before == load(REPO / "specs" / "datagen" / "fridge_base50.yaml").hash
