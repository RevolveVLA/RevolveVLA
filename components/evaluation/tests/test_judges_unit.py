"""judge c0 (plugins/judges), the judge default, the pipeline's call."""
import shutil
from pathlib import Path

import pytest
import yaml

from evaluation.core import spec as S
from evaluation.core.errors import EvalError
from evaluation.core.pipeline import Pipeline, PluginSet
from evaluation.core.registry import REGISTRY
from evaluation.core.task import load_task

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[2]
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


def parse(root, task="open_fridge_door", **edits):
    (root / "train.yaml").write_text(yaml.safe_dump(TRAIN | {"task": task}))
    raw = {"train": "train.yaml", "out": "run", "sets": {"h": {"count": 1, "pool": {"initial": [100, 100]}}}} | edits
    return S.parse(raw, "eval.yaml", root)


def test_the_default_judge_is_c0_whatever_the_task(root):
    assert parse(root).judge.plugin == "c0" and parse(root, "no_such_task").judge.plugin == "c0"
    assert parse(root, judge={"criterion": "c0"}).judge.plugin == "c0"


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


def test_c0_scores_the_evaluators_verdict_and_takes_no_parameters():
    assert judge("c0").score({"passed": True}, Path("x"), None).passed
    assert not judge("c0").score({"passed": False}, Path("x"), None).passed
    with pytest.raises(EvalError, match="takes no parameters"):
        judge("c0", ("x", 1))
