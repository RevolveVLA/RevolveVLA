"""The pieces of the live path: the sidecar files, the leakage guard, the policy under the recorder, the test policies, the client
side adapter, the robot state source, the moveit takeover's note, the spec's live keys."""
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import fake_live  # noqa: F401  (registers the fake expert)
import numpy as np
import pytest
import yaml
from datagen.core.interfaces import Config
from datagen.core.spec import SpecError
from evaluation.core import spec as E
from evaluation.core.interfaces import EvalItem
from evaluation.core.registry import REGISTRY as EVAL
from evaluation.oracle_actions import ActionSeries
from recovery import live_client
from recovery.core import flags, leakage, live
from recovery.core.spec import parse
from recovery.plugins.experts.moveit_takeover import max_delta_action
from recovery.plugins.live.ros_state import RosState

RAW = {"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": "e.yaml"}, "out": "o",
       "live": {"plugin": "dagger", "expert": {"plugin": "fake_takeover"}, "triggers": [{"plugin": "stall", "window_s": 1}]}}
TRAIN = {"task": "open_fridge_door", "robot": "mobile_panda", "generator": {"plugin": "fake_datagen"},
         "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
         "splits": {"train": {"target": 3, "pool": {"initial": [0, 9]}, "sampler": {"strategy": "stratified", "bins": {"fake_datagen.x": 3}}},
                    "aside": {"target": 2, "pool": {"initial": [20, 24]}, "record": False}}}


def eval_spec(tmp_path, **sets):
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(TRAIN | {"out": str(tmp_path / "t")}))
    raw = {"train": str(tmp_path / "train.yaml"), "out": str(tmp_path / "e"), "policy": {"plugin": "oracle"}, "sets": {
        "a": {"count": 2, "pool": {"initial": [100, 104]}}, "b": {"count": 2, "pool": {"initial": [110, 114]}},
        "tr": {"from": "train", "count": 2}, **sets}}
    return E.parse(raw, "eval.yaml", tmp_path)


# ------------------------------------------------------------------ sidecars
def test_each_episode_has_its_own_sidecar_named_after_its_result(tmp_path):
    assert flags.sidecar_of("/s/episode_3.json") == Path("/s/episode_3.interventions.json")
    flags.write(tmp_path, 1, {"phase": "grasp"})
    flags.write(tmp_path, 2, {"phase": "approach"})
    assert flags.read(tmp_path / "episode_2.json") == {"phase": "approach"} and flags.read(tmp_path / "episode_9.json") is None
    (tmp_path / "x").mkdir()
    flags.write(tmp_path / "x", 0, {"phase": "grasp"})
    (tmp_path / "x" / "episode_1.interventions.json").write_text("{broken")
    assert flags.tally(tmp_path) == {"grasp": 2, "approach": 1}


# ------------------------------------------------------------------ the leakage guard
def test_default_sets_are_the_non_heldout_ones_and_the_heldout_stay_unseen(tmp_path):
    spec = eval_spec(tmp_path)
    assert leakage.sets_of(spec, None, None) == (["tr"], ["a", "b"])
    assert leakage.sets_of(spec, ["a"], None) == (["a"], ["b"])
    assert leakage.sets_of(spec, ["a"], ["b"]) == (["a"], ["b"])
    leakage.check(spec, ["tr", "a"], ["b"])
    with pytest.raises(SpecError, match="unknown eval set \\['zz'\\]"):
        leakage.sets_of(spec, ["zz"], None)


def test_a_set_that_is_reserved_or_overlaps_a_reserved_or_aside_pool_is_refused(tmp_path):
    spec = eval_spec(tmp_path, c={"count": 1, "pool": {"initial": [20, 29]}})
    with pytest.raises(SpecError, match="also in from.heldout"):
        leakage.check(spec, ["a"], ["a"])
    with pytest.raises(SpecError, match="set 'c'.*overlaps training split 'aside'"):
        leakage.check(spec, ["c"], ["a"])
    leakage.check(spec, ["a", "tr"], ["b"])
    touching = dataclasses.replace(spec, sets={**spec.sets, "a": dataclasses.replace(spec.sets["a"], pool=E.PoolSpec((104, 112)))})
    with pytest.raises(SpecError, match=r"set 'a' \(seeds \[104, 112\]\) overlaps held-out set 'b'"):
        leakage.check(touching, ["a"], ["b"])
    only_heldout = dataclasses.replace(spec, sets={"a": spec.sets["a"]})
    with pytest.raises(SpecError, match="no set that is not held-out"):
        leakage.sets_of(only_heldout, None, None)


# ------------------------------------------------------------------ the policy under the recorder
def item(seed=7, episodes=({"passed": True, "bag": "/b"},)):
    return EvalItem(Config("a", seed), Path("/s/scene.yaml"), tuple(episodes))


class Recorder:
    def closed(self, taken):
        return frozenset(p for p, n in taken.items() if n >= 1)


def test_the_policy_client_gets_the_driver_the_closed_phases_and_the_scene(tmp_path):
    flags.write(tmp_path, 0, {"phase": "approach"})
    spec = parse(RAW)
    inner = EVAL.create("policy", "oracle", None, {})
    p = live.LivePolicy(inner, spec, Recorder(), tmp_path)
    args = p.client_args(item())
    assert args[:2] == ["--bag", "/b"] and args[2:6] == ["--instance", "/s/scene.yaml", "--driver", live.DRIVER]
    payload = json.loads(args[7])
    assert args[6] == "--driver-args" and payload == {"spec": RAW, "closed": ["approach"], "perturb": None}
    assert (p.name, p.needs_ckpt, p.client_only) == ("oracle", False, True)


def test_the_test_policies_pass_their_perturbation_and_refuse_what_they_cannot_replay(tmp_path):
    import recovery.plugins.policies  # noqa: F401
    spec, frozen = parse(RAW), EVAL.create("policy", "frozen", None, {})
    noisy = EVAL.create("policy", "noisy_expert", None, {"sigma": 0.1})
    perturb = lambda pol: json.loads(live.LivePolicy(pol, spec, Recorder(), tmp_path).client_args(item(5))[7])["perturb"]
    assert perturb(frozen) == {"name": "frozen"} and perturb(noisy) == {"name": "noisy_expert", "sigma": 0.1, "seed": 5}
    with pytest.raises(Exception, match="no passed datagen episode"):
        frozen.client_args(item(episodes=()))
    with pytest.raises(Exception, match="unknown parameters"):
        EVAL.create("policy", "noisy_expert", None, {"loud": 1})
    with pytest.raises(Exception, match="takes no parameters"):
        EVAL.create("policy", "frozen", None, {"x": 1})
    with pytest.raises(SpecError, match="cannot be halted"):
        live.LivePolicy(EVAL.create("policy", "expert", SimpleNamespace(train=SimpleNamespace(raw={})), {}), spec, Recorder(), tmp_path)


def series(n=90):
    q = np.arange(n, dtype=float)[:, None] + np.array([[0.0, 1.0]])
    return ActionSeries({"arm": ["j1", "j2"]}, {"arm": q}, np.full((n, 3), 0.3))


def test_frozen_holds_the_first_action_and_noisy_expert_is_seeded_noise_on_every_action():
    from recovery.plugins.policies import frozen, noisy_expert
    s = frozen.perturb(series(), {})
    assert (s.positions["arm"] == series().positions["arm"][0]).all() and not s.base.any() and s.joints == series().joints
    a, b = noisy_expert.perturb(series(), {"sigma": 0.1, "seed": 3}), noisy_expert.perturb(series(), {"sigma": 0.1, "seed": 3})
    c = noisy_expert.perturb(series(), {"sigma": 0.1, "seed": 4})
    assert (a.positions["arm"] == b.positions["arm"]).all() and not (a.positions["arm"] == c.positions["arm"]).all()
    assert (a.positions["arm"] - series().positions["arm"]).std() == pytest.approx(0.1, rel=0.2) and a.base.shape == (90, 3)


# ------------------------------------------------------------------ the client's side
def test_the_live_the_client_gets_is_the_recorders_driver_gate_flags_and_the_policys_alteration(tmp_path, monkeypatch):
    sim = tmp_path / "sim"
    sim.mkdir()
    seen = []

    class Rec:
        def __init__(self, spec, params):
            seen.append(("made", params["k"]))

        def driver(self, drive, ctx):
            ctx.gate.check = lambda: True
            seen.append(("closed", sorted(ctx.closed), ctx.task, ctx.robot, ctx.out, ctx.instance))
            return lambda n: {**drive(n), "x": 1}

        def finish(self, episode):
            return {"episode": episode["episode"]}
    from recovery.core.registry import REGISTRY
    REGISTRY.register("live", "fake_rec")(Rec)
    raw = {**RAW, "live": {"plugin": "fake_rec", "k": 5}}
    client = SimpleNamespace(out=str(sim))
    lv = live_client.make(client, {"spec": raw, "closed": ["open"], "perturb": {"name": "frozen"}}, "/i/scene.yaml", tmp_path)
    assert lv.halt() is False
    assert lv.driver(lambda n: {"steps": n})(4) == {"steps": 4, "x": 1} and lv.halt() is True
    assert seen == [("made", 5), ("closed", ["open"], "open_fridge_door", "mobile_panda", sim, Path("/i/scene.yaml"))]
    assert (lv.series(series()).positions["arm"] == series().positions["arm"][0]).all()
    lv.finish({"episode": 3})
    assert json.loads((sim / "episode_3.interventions.json").read_text()) == {"episode": 3}
    plain = live_client.make(client, {"spec": raw}, "/i/s.yaml", tmp_path)
    assert plain.series("same") == "same"


# ------------------------------------------------------------------ the robot's state
def test_ros_state_reads_the_joints_and_the_robots_pose_from_the_topics():
    subs = {}

    class Node:
        def create_subscription(self, kind, topic, callback, depth):
            subs[topic] = (kind, callback)
    s = RosState(Node(), "bot", types=("JointState", "TFMessage"))
    assert set(subs) == {"/joint_states", "/model/bot/pose"} and s.joints() is None and s.base() is None
    subs["/joint_states"][1](SimpleNamespace(name=["j1", "j2"], position=[0.1, 0.2]))
    assert s.joints() == {"j1": 0.1, "j2": 0.2}
    yaw = 0.8
    tf = lambda child, x: SimpleNamespace(child_frame_id=child, transform=SimpleNamespace(
        translation=SimpleNamespace(x=x, y=2.0), rotation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))))
    subs["/model/bot/pose"][1](SimpleNamespace(transforms=[tf("bot", 1.5), tf("other", 9.0)]))
    assert s.base() == pytest.approx((1.5, 2.0, yaw))
    subs["/model/bot/pose"][1](SimpleNamespace(transforms=[tf("x", 4.0)]))                 # no child of that name: the last one
    assert s.base()[0] == 4.0


def test_the_takeover_note_is_the_experts_first_second_smoothness():
    notes = json.dumps([{"what": "resume"}, {"what": "first_second", "max_delta_action": 0.027}])
    assert max_delta_action(notes) == 0.027 and max_delta_action("error: x") is None and max_delta_action(None) is None
    assert max_delta_action(json.dumps([{"what": "resume"}])) is None


# ------------------------------------------------------------------ the spec
def test_a_spec_has_phases_live_or_both_and_names_the_sets_it_rolls_out():
    assert parse(RAW).phases is None and parse(RAW).eval_spec and parse(RAW).eval_sets is None
    with pytest.raises(SpecError, match="needs `phases`"):
        parse({k: v for k, v in RAW.items() if k != "live"})
    s = parse({**RAW, "from": {"eval": "runs/eval/x", "sets": ["a"], "heldout": ["b"]}})
    assert (s.eval_spec, s.eval_sets, s.heldout) == (False, ("a",), ("b",))
    with pytest.raises(SpecError, match="from.sets: expected a list"):
        parse({**RAW, "from": {"eval": "e.yaml", "sets": "a"}})
    with pytest.raises(SpecError, match="from.bogus: unknown key"):
        parse({**RAW, "from": {"eval": "e.yaml", "bogus": 1}})
    assert parse(RAW).hash == parse({**RAW, "out": "elsewhere"}).hash != parse({**RAW, "from": {"eval": "e.yaml", "sets": ["a"]}}).hash
