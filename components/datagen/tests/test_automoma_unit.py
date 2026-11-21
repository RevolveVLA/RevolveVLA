"""plugins/automoma: the Generator adapter and the clearance Validator with the docker jobs and the scene stubbed (no
docker, no GPU), the lazy imports, and the layout the randomisers realise."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import layout_check
import specs_bridge
from datagen.core import registry, spec as S
from datagen.core.interfaces import Config, Generator, Validator
from datagen.core.pipeline import Pipeline, PluginSet
from datagen.plugins import generate_module
from datagen.plugins.automoma import adapter, clearance
from datagen.plugins.automoma.layout import Realiser
from test_generate_unit import Stub

BOX = tuple(specs_bridge.starts("open_fridge_door", "mobile_panda").centre)


def make_spec(randomisers=None, generator=None):
    return S.parse({"task": "open_fridge_door", "robot": "mobile_panda", "out": "unused",
                    "generator": {"plugin": "automoma", **(generator or {})},
                    "randomisers": randomisers if randomisers is not None else {
                        "robot_base": {"x_m": [-0.1, 0.1], "yaw_deg": [-10, 10]}, "grasp": {"id": [0, 4]}},
                    "splits": {"train": {"target": 2, "sampler": {"strategy": "random"}, "pool": {"initial": [0, 9]}}}})


def config(seed=3, **values):
    return Config("train", seed, (), {"robot_base.x_m": 0.05, "robot_base.yaw_deg": 5.0, "grasp.id": 2} | values)


class PlanStub(Stub):
    """Stub whose plan run has a starts record and a base row 0 per attempt (x = box + 0.05 + 0.001 a)."""

    def plan(self, task, seed, out, job_dir, attempts=None, layout=None):
        rc = super().plan(task, seed, out, job_dir, attempts, layout)
        h = json.loads((out / "handoff.json").read_text())
        h["starts"] = {"box_centre_obj": list(BOX), "facing_yaw_obj": np.pi, "centres_obj": []}
        (out / "handoff.json").write_text(json.dumps(h))
        traj = np.zeros((5, 36, 14))
        traj[:, 0, 0] = [BOX[0] + 0.05 + 0.001 * a for a in range(5)]
        traj[:, 0, 1] = BOX[1] - 0.02
        traj[:, 0, 2] = np.pi + np.radians(6.0)
        np.savez(out / "handoff.npz", **dict(np.load(out / "handoff.npz")), traj_robot_obj=traj)
        return rc


@pytest.fixture
def automoma(tmp_path):
    def build(stub=None, generator=None, randomisers=None):
        stub = stub or PlanStub()
        return adapter.AutoMoma(make_spec(randomisers, generator), (generator or {}), jobs=stub.jobs()), stub
    return build


def test_registered_as_generator_and_validator():
    assert issubclass(registry.get("generator", "automoma"), Generator)
    assert issubclass(registry.get("validator", "automoma"), Validator)


def test_importing_the_plugin_needs_no_heavy_stack():
    code = ("import sys; sys.path.insert(0, 'components/datagen/src'); import datagen.plugins.automoma\n"
            "bad = [m for m in ('torch', 'numpy', 'docker', 'curobo', 'automoma', 'generate') if m in sys.modules]\n"
            "assert not bad, bad")
    root = generate_module.COMPONENT.parents[1]
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_params_are_checked():
    for bad in ({"attemps": 5}, {"checks": 0}, {"checks": "3"}):
        with pytest.raises(adapter.AutoMomaError):
            adapter.AutoMoma(make_spec(), bad)
    adapter.AutoMoma(make_spec(), {"attempts": 50, "checks": 3})


def test_the_layout_is_what_the_randomisers_realise():
    r = Realiser(make_spec())
    x, y, yaw = r.layout(config())["start_pose"]
    assert (x, y, yaw) == pytest.approx((BOX[0] + 0.05, BOX[1], np.pi + np.radians(5.0)))
    assert r.layout(config())["grasp_id"] == 2
    assert Realiser(make_spec({"grasp": {"id": [0, 4]}})).layout(config()) == {"grasp_id": 2}
    assert Realiser(make_spec({})).layout(config()) == {}


def test_generate_plans_from_the_layout_and_checks_the_attempts(automoma, tmp_path):
    gen, st = automoma(PlanStub(planned=(1, 3), rejected=(1,)), {"checks": 2})
    res = gen.generate(config(), tmp_path / "seed_3")
    assert st.calls == [("plan", 3), ("check", 3, 1), ("check", 3, 3), ("check", 3, 3)]      # 1 rejected, 3: 2 of 2
    assert st.layouts[0]["grasp_id"] == 2 and st.layouts[0]["start_pose"][:2] == pytest.approx([BOX[0] + 0.05, BOX[1]])
    assert res.ok and res.realised["attempt_id"] == 3
    assert res.realised["robot_base.x_m"] == pytest.approx(0.05 + 0.003) and res.realised["grasp_id"] == 2
    assert res.realised["robot_base.y_m"] == pytest.approx(-0.02) and res.realised["robot_base.yaw_deg"] == pytest.approx(6.0)
    assert res.realised["start_pose"] == pytest.approx([BOX[0] + 0.053, BOX[1] - 0.02, np.pi + np.radians(6.0)])
    assert {k: v.replace(str(tmp_path), "") for k, v in res.paths.items()} == {
        "plan": "/seed_3/plans/seed_3", "check": "/seed_3/checks/seed_3_a3", "manifest": "/seed_3/manifest.json"}
    assert not (tmp_path / "seed_3" / "episodes").exists()               # nothing is recorded here
    assert res.to_dict()["ok"] and json.dumps(res.to_dict())              # plain JSON for the manifest


def test_generate_without_a_varied_start_plans_the_clustered_way(automoma, tmp_path):
    gen, st = automoma(PlanStub(planned=(0,)), randomisers={})
    assert gen.generate(Config("train", 1), tmp_path / "e").ok and st.layouts == [None]


def test_generate_failures_say_why(automoma, tmp_path):
    cases = {"plan failed": dict(plan_rc=3), "FK check failed": dict(plan_rc=1), "replay rejected": dict(rejected=(0, 1, 2))}
    for reason, kw in cases.items():
        gen, _ = automoma(PlanStub(**kw))
        res = gen.generate(config(), tmp_path / reason.replace(" ", "_"))
        assert not res.ok and res.reason == reason and res.realised == {}


def test_generate_failure_names_the_repeat_check_that_failed(automoma, tmp_path):
    gen, _ = automoma(PlanStub(planned=(0,), dirs={"seed_3_a0_r1": "reject"}), {"checks": 2})
    res = gen.generate(config(), tmp_path / "e")
    assert not res.ok and res.reason == "replay rejected (repeat check 2/2 failed)"


def test_generate_reports_an_infrastructure_error(automoma, tmp_path):
    gen, _ = automoma(PlanStub(dirs={"seed_3_a0": "crash"}))
    res = gen.generate(config(), tmp_path / "e")
    assert not res.ok and "docker died" in res.reason


def test_generate_sets_the_robot_and_resumes(automoma, tmp_path, monkeypatch):
    monkeypatch.delenv("REVOLVE_ROBOT")
    gen, st = automoma()
    assert gen.generate(config(), tmp_path / "e").ok
    assert os.environ["REVOLVE_ROBOT"] == "mobile_panda"
    n = len(st.calls)
    assert gen.generate(config(), tmp_path / "e").ok and len(st.calls) == n      # generate.py's manifest: nothing redone


def test_a_record_false_split_ends_with_its_realised_starts(tmp_path):
    raw = {"task": "open_fridge_door", "robot": "mobile_panda", "out": "unused", "generator": {"plugin": "automoma"},
           "randomisers": {"robot_base": {"x_m": [-0.1, 0.1]}},
           "splits": {"heldout": {"target": 2, "sampler": {"strategy": "random"}, "pool": {"initial": [10, 14]},
                                  "record": False}}}
    spec, stub = S.parse(raw), PlanStub(planned=(0, 1), rejected=(0,))
    plugins = PluginSet(samplers={"random": registry.create("sampler", "random")},
                        randomisers={"robot_base": registry.create("randomiser", "robot_base")},
                        generator=adapter.AutoMoma(spec, {}, jobs=stub.jobs()),
                        recorder=pytest.fail)                           # never called: record is false
    assert Pipeline(spec, plugins, tmp_path, log=lambda m: None).run(["sample", "generate", "record"])
    used = json.loads((tmp_path / "manifest.json").read_text())["splits"]["heldout"]["used"]
    assert [u["config"]["seed"] for u in used] == [10, 11] and all(u["generate"]["ok"] and "record" not in u for u in used)
    for u in used:
        x = u["config"]["values"]["robot_base.x_m"]
        assert u["generate"]["realised"]["attempt_id"] == 1 and u["generate"]["realised"]["robot_base.x_m"] == pytest.approx(0.051)
        assert -0.1 <= x <= 0.1 and (tmp_path / "heldout" / "episodes" / f"seed_{u['config']['seed']}").is_dir()


def test_clearance_validator_rejects_with_the_reason(monkeypatch):
    seen = []
    monkeypatch.setattr(layout_check, "clearance_reject",
                        lambda task, robot: lambda layout: seen.append((task, robot, layout)) or "collision: chassis 0.0 m")
    v = clearance.ChassisClearance(make_spec(), {})
    res = v.validate(config())
    assert not res.ok and res.reason.startswith("collision") and seen[0][:2] == ("open_fridge_door", "mobile_panda")
    assert seen[0][2]["start_pose"][0] == pytest.approx(BOX[0] + 0.05)
    monkeypatch.setattr(layout_check, "clearance_reject", lambda task, robot: lambda layout: None)
    assert clearance.ChassisClearance(make_spec(), {}).validate(config()).ok


ASSETS = layout_check.specs_bridge.ROOT / "assets" / "generated_partnet" / "fridge" / "model.sdf"


@pytest.mark.skipif(not ASSETS.is_file(), reason="scene assets (assets/, not in git) are absent")
def test_clearance_validator_on_the_real_scene():
    v = clearance.ChassisClearance(make_spec(), {})
    assert v.validate(config()).ok and v.validate(config(**{"robot_base.x_m": -0.17, "robot_base.yaw_deg": 10})).ok
