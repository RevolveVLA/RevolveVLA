"""core/spec.py: parsing, defaults, errors naming the yaml path, the hash, the datagen spec of the generator."""
import copy
from pathlib import Path

import pytest
import yaml

from datagen.core.spec import SpecError
from recovery.core import spec as S

RAW = {"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": "runs/eval/x"}, "out": "runs/recovery/x",
       "phases": {"generator": {"plugin": "moveit", "attempts": 4, "record": "no-images"}, "quota": {"approach": 3, "grasp": 2}}}


def parse(**edits):
    raw = copy.deepcopy(RAW)
    for path, value in edits.items():
        d = raw
        *keys, last = path.split("__")
        for k in keys:
            d = d[k]
        d[last] = value
    return S.parse(raw, "r.yaml")


def test_defaults():
    s = parse()
    assert (s.eval_run, s.keep, s.seed, s.live) == ("runs/eval/x", "successes", 0, None)
    assert s.triggers == ({"plugin": "verdict"},)
    p = s.phases
    assert (p.restart_from, p.back_s, p.max_object_joint, p.per_failure) == ("before_failure", 1.0, 0.02, 1)
    assert (p.generator.plugin, p.episodes.attempts, p.episodes.record, p.quota) == ("moveit", 4, "no-images", {"approach": 3, "grasp": 2})
    assert s.run == S.RunSpec(1, "relaunch", True) and (s.runner.plugin, s.scanner.plugin) == ("sim", "container")


def test_full_spec_of_the_repo_parses():
    for name in ("fridge", "smoke", "fridge_coupled"):
        s = S.load(f"{__file__.rsplit('/components', 1)[0]}/specs/recovery/{name}.yaml")
        assert s.task == "open_fridge_door" and s.phases.quota


def test_the_coupled_example_has_phases_and_a_live_recorder_and_its_evaluation_names_it():
    import dataclasses
    from evaluation.core import spec as E
    from recovery.core import coupling
    from recovery.core.scan import checked
    repo = Path(__file__).resolve().parents[3]
    ev = E.load(repo / "specs/eval/fridge_recovery.yaml", repo)
    assert ev.coupled and ev.recovery.spec == "specs/recovery/fridge_coupled.yaml" and ev.run.reset == "warm"
    spec = checked(coupling.coupled_spec(S.load(repo / ev.recovery.spec), ev, repo / "runs/eval/fridge_recovery"))
    assert spec.phases and spec.live.plugin == "dagger" and dataclasses.astuple(spec.run) == (2, "warm", True)


def test_per_failure_is_the_episodes_per_scene_of_the_datagen_spec():
    s = parse(phases__per_failure=3, phases__generator={"plugin": "moveit", "attempts": 5, "extra": 1})
    d = s.datagen_spec()
    assert (d.task, d.robot, d.generator.plugin, dict(d.generator.params)) == ("open_fridge_door", "mobile_panda", "moveit", {"extra": 1})
    assert (d.episodes.per_scene, d.episodes.attempts) == (3, 5)
    assert s.phases.per_failure == 3


@pytest.mark.parametrize("edit, message", [
    ({"phases__restart_from": "never"}, "phases.restart_from: expected one of"),
    ({"phases__back_s": -1}, "phases.back_s: must be >= 0"),
    ({"phases__quota": {"global": 2}}, "phases.quota.global: a global"),
    ({"phases__quota": {"approach": -1}}, "phases.quota.approach: must be >= 0"),
    ({"phases__generator": {"plugin": "m", "episodes_per_scene": 2}}, "phases.generator.episodes_per_scene: use r.yaml.phases.per_failure"),
    ({"phases__per_failure": 3, "phases__generator": {"plugin": "m", "attempts": 2}}, "cannot come from 2 attempts"),
    ({"keep": "some"}, "keep: expected one of"),
    ({"run": {"reset": "hot"}}, "run.reset: expected one of"),
    ({"run": {"workers": 0}}, "run.workers: must be >= 1"),
    ({"triggers": []}, "triggers: expected a non-empty list"),
    ({"triggers": [{"nope": 1}]}, None),
    ({"from": {"nope": "x"}}, "from"),
    ({"bogus": 1}, "bogus: unknown key"),
])
def test_errors_name_the_yaml_path(edit, message):
    with pytest.raises(SpecError) as e:
        parse(**edit)
    if message:
        assert message in str(e.value)


def test_missing_keys():
    with pytest.raises(SpecError, match="missing key 'quota'"):
        S.parse({**RAW, "phases": {"generator": {"plugin": "m"}}}, "r.yaml")


def test_hash_ignores_out_and_run_only():
    base = parse().hash
    assert parse(out="elsewhere", run={"workers": 5, "reset": "warm", "headless": False}).hash == base
    for edit in ({"phases__quota": {"approach": 4}}, {"keep": "all"}, {"seed": 1}, {"phases__back_s": 2.0},
                 {"triggers": [{"plugin": "verdict"}, {"plugin": "stall"}]}, {"from": {"eval": "runs/eval/y"}}):
        assert parse(**edit).hash != base


def test_load_reports_unreadable_files(tmp_path):
    with pytest.raises(SpecError, match="cannot read"):
        S.load(tmp_path / "missing.yaml")
    (tmp_path / "bad.yaml").write_text("a: [")
    with pytest.raises(SpecError, match="cannot read"):
        S.load(tmp_path / "bad.yaml")
    (tmp_path / "ok.yaml").write_text(yaml.safe_dump(RAW))
    assert S.load(tmp_path / "ok.yaml").task == "open_fridge_door"


def test_check_task_rejects_unknown_phases():
    S.check_task(parse(), ["approach", "grasp", "open"])
    with pytest.raises(SpecError, match=r"phases.quota: unknown phase \['grasp'\]"):
        S.check_task(parse(), ["approach"])
