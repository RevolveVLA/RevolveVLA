"""plugins/rollouts (docker over a monkeypatched rollout.run, fake) and plugins/policies (CPU, no docker), and the
runner params rollout.py writes."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from datagen.core.interfaces import Config

from evaluation.core.errors import EvalError
from evaluation.core.interfaces import EvalItem, ResolvedCkpt, Rollout
from evaluation.core.registry import REGISTRY
from evaluation.core.spec import RunnerSpec
from evaluation.plugins.rollouts import docker as D

COMP = Path(__file__).resolve().parents[1]
spec = SimpleNamespace(train=SimpleNamespace(task="open_fridge_door", robot="mobile_panda"))
smolvla, oracle = (REGISTRY.create("policy", n, spec, {}) for n in ("smolvla", "oracle"))


@pytest.fixture
def item(tmp_path):
    return EvalItem(Config("heldout", 501), tmp_path / "start")


def test_runner_params_yaml_default_and_override():
    import rollout as R
    base = yaml.safe_load(R.runner_params_yaml())["policy_runner"]["ros__parameters"]
    assert base == R.RUNNER_PARAMS and base["chunk_size_threshold"] == 0.95          # the manager keeps blending
    over = yaml.safe_load(R.runner_params_yaml(dict(actions_per_chunk=21, chunk_size_threshold=0.0)))
    assert over["policy_runner"]["ros__parameters"] == dict(R.RUNNER_PARAMS, actions_per_chunk=21, chunk_size_threshold=0.0)


def test_runner_params_yaml_aggregate():
    import rollout as R
    get = lambda **o: yaml.safe_load(R.runner_params_yaml(o))["policy_runner"]["ros__parameters"]
    for runner, want in [(RunnerSpec(11, 0.0), dict(actions_per_chunk=11, chunk_size_threshold=0.0, aggregate_fn_name="weighted_average")),
                         (RunnerSpec(21, 0.5, "weighted_average"), dict(actions_per_chunk=21, chunk_size_threshold=0.5,
                                                                         aggregate_fn_name="weighted_average")),
                         (RunnerSpec(21, 0.5, "latest_only"), dict(actions_per_chunk=21, chunk_size_threshold=0.5,
                                                                    aggregate_fn_name="latest_only"))]:
        assert get(**runner.params()) == want


def test_policies():
    assert (smolvla.name, smolvla.needs_ckpt, smolvla.client_only) == ("smolvla", True, False)
    assert (oracle.name, oracle.needs_ckpt, oracle.client_only) == ("oracle", False, True)
    with pytest.raises(EvalError, match="no parameters"):
        REGISTRY.create("policy", "oracle", spec, {"x": 1})


def test_docker_passes_the_rollout_run_kwargs(monkeypatch, item, tmp_path):
    calls = []
    monkeypatch.setattr(D, "load", lambda name="rollout": SimpleNamespace(run=lambda *a, **k: calls.append((a, k)) or {"passed": True}))
    r = REGISTRY.create("rollout", "docker", spec, {"budget_s": 6, "rtf": 0.2})
    ckpt = ResolvedCkpt(str(tmp_path / "ckpt"), "x")
    rec = r.run(item, tmp_path / "out", seed=501, runner=RunnerSpec(21, 0.0), policy=smolvla, ckpt=ckpt, gui=True,
                restore=(tmp_path / "prev", "open"))
    assert rec == {"passed": True}                                                       # the record, unchanged
    (args, kw), = calls
    assert args == (item.scene, tmp_path / "out")
    assert kw == dict(seed=501, policy="smolvla", ckpt=ckpt.path, restore=(tmp_path / "prev", "open"), gui=True,
                      task="open_fridge_door", runner_params=dict(actions_per_chunk=21, chunk_size_threshold=0.0),
                      budget_s=6, rtf=0.2)
    r.run(item, tmp_path / "o3", seed=1, runner=RunnerSpec(11, 0.5, "average"), policy=smolvla, ckpt=ckpt)
    assert calls[1][1]["runner_params"] == dict(actions_per_chunk=11, chunk_size_threshold=0.5, aggregate_fn_name="average")
    calls.pop()
    r.run(item, tmp_path / "o2", seed=1, runner=RunnerSpec(), policy=oracle, ckpt=None)
    assert calls[1][1]["policy"] == "oracle" and calls[1][1]["ckpt"] is None and calls[1][1]["restore"] is None


def test_docker_sets_the_robot_env_for_rollout_py(monkeypatch):
    monkeypatch.delenv("REVOLVE_ROBOT", raising=False)
    REGISTRY.create("rollout", "docker", spec, {})
    assert os.environ["REVOLVE_ROBOT"] == "mobile_panda"            # rollout.py loads the robot-free task through it
    monkeypatch.delenv("REVOLVE_ROBOT")


def test_docker_rejects_unknown_params():
    with pytest.raises(EvalError, match="unknown parameters.*speed"):
        REGISTRY.create("rollout", "docker", spec, {"speed": 2})


def test_docker_default_workers_is_the_gpu_default(monkeypatch):
    seen = []
    monkeypatch.setattr(D, "load", lambda name="rollout": SimpleNamespace(default_parallel=lambda p: seen.append((name, p)) or 3))
    assert REGISTRY.create("rollout", "docker", spec, {}).default_workers(oracle) == 3 and seen == [("evaluate", "oracle")]
    assert REGISTRY.create("rollout", "fake", spec, {}).default_workers(oracle) == 1


def test_docker_plugin_imports_no_docker():
    code = ("import sys; sys.path[:0] = ['%s', '%s']; import evaluation.plugins.rollouts.docker as D; "
            "assert not {'rollout', 'docker'} & set(sys.modules), sorted(sys.modules)" % (COMP / "src", COMP.parent / "datagen" / "src"))
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fake_contract_and_scripted_verdicts(item, tmp_path):
    r = REGISTRY.create("rollout", "fake", spec, {"fail_seeds": [501], "failed_phase": "open"})
    assert isinstance(r, Rollout)
    run = lambda out, **kw: r.run(item, tmp_path / out, seed=501, runner=RunnerSpec(), policy=smolvla, ckpt=None, **kw)
    rec = run("a")
    assert rec["passed"] is False and rec["failed_phase"] == "open" and rec["failed_checks"] == ["open"]
    assert rec["phases"] == {"approach": "pass", "grasp": "pass", "open": "fail", "release": "not_reached"}
    d = tmp_path / "a"
    assert json.loads((d / "rollout.json").read_text()) == rec
    assert json.loads((d / "phase_report.json").read_text())["result"]["failed_phase"] == "open"
    assert sorted(p.name for p in (d / "phase_states").iterdir()) == [f"{p}_start.npz" for p in ("approach", "grasp", "open")]
    retry = run("b", restore=(d, "open"))                                                 # a retry from the saved state passes
    assert retry["passed"] and retry["restore"] == {"source": str(d), "phase": "open"}
    always = REGISTRY.create("rollout", "fake", spec, {"fail_seeds": [501], "always_fail": True})
    assert not always.run(item, tmp_path / "e", seed=501, runner=RunnerSpec(), policy=smolvla, ckpt=None, restore=(d, "x"))["passed"]
    other = r.run(item, tmp_path / "f", seed=502, runner=RunnerSpec(), policy=smolvla, ckpt=None)
    assert other["passed"] and other["failed_phase"] is None
