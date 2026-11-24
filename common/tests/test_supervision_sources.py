"""LiveSource (a SimClient fed by hand) and BagSource (a fake bag reader)."""
import json

import pytest

from conftest import SPECS
from revolve_common import specs
from revolve_common.sim_client import SimClient
from revolve_common.supervision import BagMessage, BagSource, LiveSource, build

ROBOT = "mobile_panda"


def state(t, phase="approach", episode=0, **values):
    return json.dumps({"t": t, "ready": True, "resets": 0, "episode": episode, "phase": phase, "values": values})


def test_live_source_ticks_from_the_newest_state():
    c = SimClient()
    arm, base = {"panda_joint1": 0.1}, (1.0, 2.0, 0.3)
    src = LiveSource(c, joints=lambda: arm, base=lambda: base)
    assert src.tick() is None                                        # no state yet
    c.on_state(json.dumps({"t": 5.0, "ready": True, "resets": 0, "episode": None, "phase": None, "values": {}}))
    assert src.tick() is None                                        # no episode running
    c.on_state(state(10.0, **{"approach.done[0]": 0.3}))
    t = src.tick()
    assert (t.t, t.phase, t.joints, t.base) == (0.0, "approach", arm, base) and t.values == {"approach.done[0]": 0.3}
    assert src.tick() is None                                        # nothing new
    c.on_state(state(10.1))
    c.on_state(state(10.2, phase="grasp"))
    t = src.tick()                                                   # the newest one
    assert (t.t, t.phase) == (pytest.approx(0.2), "grasp")


def test_live_source_time_zero_per_episode_and_no_robot_state():
    c = SimClient()
    src = LiveSource(c)
    c.on_state(state(10.0))
    assert src.tick().joints is None and src.tick() is None
    c.on_state(state(50.0, episode=1))
    assert src.tick().t == 0.0
    c.on_state(state(50.1, episode=1))
    assert src.tick().t == pytest.approx(0.1)


def test_live_ticks_drive_a_supervisor():
    c = SimClient()
    src = LiveSource(c)
    sup = build([{"plugin": "phase_time", "after_s": 0.15}], specs.resolve(SPECS, "open_fridge_door", assets=False))
    fired = None
    for t in (10.0, 10.1, 10.2):
        c.on_state(state(t))
        fired = sup.check(src.tick())
    assert fired.t == pytest.approx(0.2)


# ------------------------------------------------------------------ bag
T0 = 100.0                                  # sim time at the episode start


def msgs(passed=False, with_result=True, states=16):
    """A 1.5 s episode: 10 Hz states, 50 Hz joints, 20 Hz base pose, one scene joint; fails `grasp` at 1.5 s."""
    out = [BagMessage("/evaluator/state", T0 - 0.1, state(T0 - 0.1, phase=None, episode=None))]
    for i in range(states):
        t = T0 + 0.1 * i
        out.append(BagMessage("/evaluator/state", t, state(t, "approach" if t < T0 + 1 else "grasp", episode=3,
                                                          **{"grasp.done[0]": 2 - i * 0.1})))
    out += [BagMessage("/joint_states", T0 + 0.02 * i, {"name": ["panda_joint1", "panda_finger_joint1"],
                                                       "position": [0.01 * i, 0.04]}) for i in range(-5, 100)]
    out += [BagMessage("/model/mobile_panda/pose", T0 + 0.05 * i, (0.1 * i, 2.0, 0.0)) for i in range(-2, 40)]
    out += [BagMessage("/world/kitchen/model/fridge/joint_state", T0 + 1.0,
                       {"name": ["top_door_hinge"], "position": [0.5]})]
    out += [BagMessage("/some/image", T0, b"xx")]       # never asked for: the reader filters, a fake may not
    if with_result:
        result = {"passed": passed, "failed_phase": None if passed else "grasp",
                  "reason": None if passed else "timeout_s", "t_fail": None if passed else 1.5, "duration_s": 2.0,
                  "sim_time_start": T0, "episode": 3}
        out.append(BagMessage("/evaluator/result", T0 + 1.5, json.dumps(result)))
    return out


def reader(messages):
    seen = []

    def read(bag, wanted):
        seen.append(bag)
        return [m for m in messages if wanted(m.topic)]
    read.seen = seen
    return read


def test_bag_source_ticks_are_episode_relative_and_carry_the_robot_state():
    r = reader(msgs())
    src = BagSource(r, "bag_dir", ROBOT)
    ticks = list(src.ticks())
    assert r.seen == ["bag_dir"]
    assert [round(t.t, 1) for t in ticks[:3]] == [0.0, 0.1, 0.2] and ticks[0].phase == "approach"
    t = ticks[10]
    assert t.t == pytest.approx(1.0) and t.phase == "grasp" and t.values["grasp.done[0]"] == pytest.approx(1.0)
    assert t.joints == {"panda_joint1": pytest.approx(0.5), "panda_finger_joint1": 0.04}      # 50 Hz: i=50
    assert t.base == (pytest.approx(2.0), 2.0, 0.0)
    assert len(ticks) == 17 and ticks[-1].result["failed_phase"] == "grasp"
    assert (ticks[-1].t, ticks[-1].phase) == (1.5, "grasp") and ticks[-1].t >= ticks[-2].t


def test_bag_source_last_tick_never_before_the_last_state():
    last = list(BagSource(reader(msgs(states=21)), "b", ROBOT).ticks())[-1]      # states run on to 2.0 s
    assert last.t == pytest.approx(2.0) and last.result


def test_bag_source_passed_episode_has_no_result_tick():
    ticks = list(BagSource(reader(msgs(passed=True)), "b", ROBOT).ticks())
    assert len(ticks) == 16 and all(t.result is None for t in ticks)


def test_bag_source_given_result_wins_over_the_bag():
    given = {"passed": False, "failed_phase": "open", "reason": "while", "check": "open.while[1]", "t_fail": 0.7,
             "sim_time_start": T0}
    src = BagSource(reader(msgs(with_result=False)), "b", ROBOT, result=given)
    assert src.result is given and list(src.ticks())[-1].result is given


def test_bag_source_time_zero_falls_back_to_the_first_episode_state():
    src = BagSource(reader(msgs(with_result=False)), "b", ROBOT)
    assert src.t0 == T0 and src.result is None
    assert list(src.ticks())[0].t == 0.0


def test_state_at_is_the_last_message_at_or_before_t():
    src = BagSource(reader(msgs()), "b", ROBOT)
    s = src.state_at(0.51)
    assert s.t == 0.51 and s.joints["panda_joint1"] == pytest.approx(0.25)          # message at 0.50 s (i = 25)
    assert s.base == (pytest.approx(1.0), 2.0, 0.0)                                  # 0.50 s: i = 10
    assert s.objects == {}                                                           # the door joint starts at 1.0 s
    assert src.state_at(1.0).objects == {"fridge::top_door_hinge": 0.5}
    assert src.state_at(1.99).objects == {"fridge::top_door_hinge": 0.5}
    exact = src.state_at(0.5)
    assert exact.joints["panda_joint1"] == pytest.approx(0.25)                       # at, not after
    assert src.state_at(-1.0).joints == {} and src.state_at(-1.0).base is None


def test_bag_source_with_a_supervisor_end_to_end():
    src = BagSource(reader(msgs()), "b", ROBOT)
    sup = build([{"plugin": "condition", "phase": "grasp", "value": "grasp.done[0]", "op": "lt", "threshold": 0.55},
                 {"plugin": "verdict"}], specs.resolve(SPECS, "open_fridge_door", assets=False))
    f = sup.scan(src.ticks())
    assert (f.trigger, f.phase) == ("condition", "grasp") and f.t == pytest.approx(1.5)
    assert dict(sup.history) == {"condition": 1}
    v = build([{"plugin": "verdict"}], specs.resolve(SPECS, "open_fridge_door", assets=False)).scan(src.ticks())
    assert v.reason == "grasp:timeout_s" and v.t == pytest.approx(1.5)


def test_unsorted_messages_are_ordered():
    m = msgs()
    src = BagSource(reader(list(reversed(m))), "b", ROBOT)
    assert [t.t for t in src.ticks()] == sorted(t.t for t in src.ticks())
    assert src.state_at(0.51).joints["panda_joint1"] == pytest.approx(0.25)


def test_first_state_t_is_when_a_bag_that_starts_late_has_base_and_joints():
    late = [m for m in msgs() if m.topic not in ("/joint_states", "/model/mobile_panda/pose") or m.t >= T0 + 0.07]
    src = BagSource(reader(late), "b", ROBOT)
    assert src.first_state_t == pytest.approx(0.1) and src.state_at(0.0).base is None
    assert src.state_at(src.first_state_t).base is not None and src.state_at(src.first_state_t).joints
    assert BagSource(reader([m for m in msgs() if m.topic != "/joint_states"]), "b", ROBOT).first_state_t is None
