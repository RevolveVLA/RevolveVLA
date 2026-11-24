"""plugins/rollouts/fake and plugins/policies (CPU, no sim)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from datagen.core.interfaces import Config

from evaluation.core.errors import EvalError
from evaluation.core.interfaces import EvalItem, Rollout
from evaluation.core.registry import REGISTRY
from evaluation.core.spec import RunnerSpec

spec = SimpleNamespace(train=SimpleNamespace(task="open_fridge_door", robot="mobile_panda"))
smolvla, oracle = (REGISTRY.create("policy", n, spec, {}) for n in ("smolvla", "oracle"))


@pytest.fixture
def item(tmp_path):
    return EvalItem(Config("heldout", 501), tmp_path / "scene.yaml")


def test_policies():
    assert (smolvla.name, smolvla.needs_ckpt, smolvla.client_only) == ("smolvla", True, False)
    assert (oracle.name, oracle.needs_ckpt, oracle.client_only) == ("oracle", False, True)
    with pytest.raises(EvalError, match="no parameters"):
        REGISTRY.create("policy", "oracle", spec, {"x": 1})


def test_the_oracle_replays_the_first_passed_bag_of_the_config(tmp_path):
    eps = ({"passed": False, "bag": "/b/0"}, {"passed": True, "bag": None}, {"passed": True, "bag": "/b/2"}, {"passed": True, "bag": "/b/3"})
    assert oracle.client_args(EvalItem(Config("train", 1), tmp_path, eps)) == ["--bag", "/b/2"]
    with pytest.raises(EvalError, match="no passed datagen episode with a bag for seed 1"):
        oracle.client_args(EvalItem(Config("train", 1), tmp_path, eps[:2]))


def test_fake_contract_and_scripted_verdicts(item, tmp_path):
    r = REGISTRY.create("rollout", "fake", spec, {"fail_seeds": [501], "failed_phase": "open"})
    assert isinstance(r, Rollout)
    run = lambda out, seed=501: r.run(item, tmp_path / out, seed=seed, runner=RunnerSpec(), policy=smolvla, ckpt=None)
    rec = run("a")
    assert rec["passed"] is False and rec["failed_phase"] == "open" and rec["reason"] == "timeout_s" and rec["error"] is None
    assert rec["phases"] == {"approach": "passed", "grasp": "passed", "open": "failed"}
    assert json.loads((tmp_path / "a" / "rollout.json").read_text()) == rec
    assert run("retry_1")["passed"]                                   # a retry is a new episode from the start: it passes
    always = REGISTRY.create("rollout", "fake", spec, {"fail_seeds": [501], "always_fail": True})
    assert not always.run(item, tmp_path / "retry_2", seed=501, runner=RunnerSpec(), policy=smolvla, ckpt=None)["passed"]
    other = run("f", seed=502)
    assert other["passed"] and other["failed_phase"] is None and other["reason"] is None


def test_fake_end_reasons_and_lifecycle(item, tmp_path):
    r = REGISTRY.create("rollout", "fake", spec, {"end_reasons": {1: "while", 2: "wall_limit", 3: "interrupted"}})
    run = lambda seed: r.run(item, tmp_path / str(seed), seed=seed, runner=RunnerSpec(), policy=smolvla, ckpt=None)
    assert (run(1)["reason"], run(1)["passed"]) == ("while", False) and run(1)["failed_phase"] == "open"
    assert run(2)["error"].startswith("wall limit") and run(2)["phases"] is None
    assert run(3)["error"].startswith("interrupted")
    r.release(item), r.stop()
    assert r.released == [501] and r.stopped and r.default_workers(smolvla) == 1
