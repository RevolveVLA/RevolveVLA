"""Hand-back of live dagger: the state machine policy -> expert -> policy -> expert, the stop condition, hysteresis (settle), the
cap, the flags and the jump measure. The one-way default is test_dagger_unit.py."""
import fake_live as L
import pytest
from datagen.core.spec import SpecError
from fake_live import LOG, Script
from recovery.core.registry import REGISTRY
from recovery.plugins.live import dagger
from recovery.plugins.live import handback as hb
from test_dagger_unit import PHASE_TIME, episode, run, spec_of

GRASP_TIME = {"plugin": "phase_time", "phase": "grasp", "after_s": 1.0}
BACK = {"when": "phase_done", "settle_s": 0.3}
EXPERT = {"plugin": "fake_takeover", "drives": True}
TWO = Script()                                          # approach 0-6 s, grasp 6-9, open 9-; the verdict at 20 s: failed


@pytest.fixture(autouse=True)
def stub_state(monkeypatch):
    LOG.clear()
    monkeypatch.setattr(dagger, "RosState", L.StateStub)


def flags_of(client, recorder):
    return recorder.finish({"episode": 0, "result": client.wait_files(0)})


def test_the_expert_hands_back_when_the_phase_has_advanced_and_the_policy_runs_on(tmp_path):
    client, recorder, ctx, drive, published = episode(tmp_path, TWO, handback=BACK, expert=EXPERT)
    extra, result = run(client, drive)
    flags = flags_of(client, recorder)
    assert not result["passed"] and len(recorder.takes) == 1 and recorder.pending is None
    (stopped,) = [x for x in LOG if isinstance(x, tuple) and x[0] == "stopped"]
    assert 6.0 <= stopped[1] < 6.3 and LOG.index(stopped) < len(LOG) - 1 and LOG[-1] == "policy"      # the policy acts again after it
    assert extra["steps"] == len(published) > 400 and extra["notes"] == "stopped"
    a, b = flags["takeover_t"], flags["interventions"][0]["handback_t"]
    assert flags["segments"] == [[0.0, a, "policy"], [a, b, "expert"], [b, 20.0, "policy"]]
    assert flags["interventions"][0]["handback_reason"] == "phase_done: approach -> grasp" and 6.0 <= b < 6.3


def test_one_episode_holds_several_interventions_each_with_its_own_flags(tmp_path):
    client, recorder, ctx, drive, _ = episode(tmp_path, TWO, triggers=[PHASE_TIME, GRASP_TIME], handback=BACK, expert=EXPERT)
    _, result = run(client, drive)
    flags = flags_of(client, recorder)
    assert [t.phase for t in recorder.takes] == ["approach", "grasp"] and recorder.taken is recorder.takes[0]
    assert [s[2] for s in flags["segments"]] == ["policy", "expert", "policy", "expert", "policy"]
    first, second = flags["interventions"]
    assert (first["phase"], first["handback_reason"]) == ("approach", "phase_done: approach -> grasp")
    assert (second["phase"], second["handback_reason"]) == ("grasp", "phase_done: grasp -> open")
    assert second["takeover_t"] >= first["handback_t"] + 0.3 + 1.0 - 0.15                       # settle, then the debounce again
    assert set(first) == {"trigger", "reason", "phase", "takeover_t", "handback_t", "handback_reason", "max_delta_action", "expert_failure"}
    assert first["max_delta_action"]["takeover"] == 0.02 and flags["trigger"] == "phase_time" and flags["takeover_t"] == first["takeover_t"]
    assert flags["expert_failure"] is False and flags["kept"] is None and flags["segment"] is None


def test_no_new_takeover_during_the_settle_time(tmp_path):
    quick = {"plugin": "phase_time", "phase": "grasp", "after_s": 0.1}
    client, recorder, ctx, drive, _ = episode(tmp_path, TWO, triggers=[PHASE_TIME, quick], handback={**BACK, "settle_s": 1.5}, expert=EXPERT)
    run(client, drive)
    first, second = flags_of(client, recorder)["interventions"]
    assert second["takeover_t"] - first["handback_t"] >= 1.5 + 0.1 - 0.15


def test_the_cap_ends_the_takeovers_and_the_policy_runs_to_the_verdict(tmp_path):
    client, recorder, ctx, drive, _ = episode(tmp_path, TWO, triggers=[PHASE_TIME, GRASP_TIME], handback=BACK, expert=EXPERT,
                                              max_interventions=1)
    run(client, drive)
    flags = flags_of(client, recorder)
    assert len(recorder.takes) == 1 and len(flags["interventions"]) == 1 and ctx.gate() is False and flags["segments"][-1][2] == "policy"


def test_after_s_stops_the_expert_that_long_after_the_takeover(tmp_path):
    client, recorder, *_, drive, _ = episode(tmp_path, TWO, handback={"when": "after_s", "after_s": 2.0}, expert=EXPERT)
    run(client, drive)
    i = flags_of(client, recorder)["interventions"][0]
    assert i["handback_t"] - i["takeover_t"] == pytest.approx(2.0, abs=0.2) and i["handback_reason"].startswith("after_s")


def test_an_expert_that_finishes_or_fails_is_not_handed_back(tmp_path):
    client, recorder, ctx, drive, _ = episode(tmp_path, handback=BACK, expert={"plugin": "fake_takeover"})        # finishes at once
    run(client, drive)
    flags = flags_of(client, recorder)
    assert flags["interventions"][0]["handback_t"] is None and flags["segments"][-1][2] == "expert" and not flags["expert_failure"]
    LOG.clear()
    client, recorder, ctx, drive, _ = episode(tmp_path, handback=BACK, expert={"plugin": "fake_takeover", "raises": True})
    run(client, drive)
    flags = flags_of(client, recorder)
    assert flags["interventions"][0]["expert_failure"] is True and flags["interventions"][0]["handback_t"] is None and flags["expert_failure"]


def test_the_policy_that_fails_after_a_handback_is_not_the_experts_failure(tmp_path):
    client, recorder, ctx, drive, _ = episode(tmp_path, TWO, handback=BACK, expert=EXPERT)          # nobody passes it
    _, result = run(client, drive)
    flags = flags_of(client, recorder)
    assert not result["passed"] and flags["expert_failure"] is False and flags["interventions"][0]["handback_t"] is not None


def test_record_interventions_keeps_every_expert_span(tmp_path):
    client, recorder, *_, drive, _ = episode(tmp_path, TWO, triggers=[PHASE_TIME, GRASP_TIME], handback=BACK, expert=EXPERT,
                                             record="interventions")
    run(client, drive)
    flags = flags_of(client, recorder)
    assert flags["kept"] == [s[:2] for s in flags["segments"] if s[2] == "expert"] and len(flags["kept"]) == 2 and flags["segment"] is None
    client, recorder, *_, drive, _ = episode(tmp_path, TWO, handback=BACK, expert=EXPERT, record="interventions")
    run(client, drive)
    flags = flags_of(client, recorder)
    assert flags["segment"] == flags["kept"][0] and len(flags["kept"]) == 1


def test_the_jump_at_the_hand_back_is_the_arms_largest_step_of_the_first_second(tmp_path):
    client, recorder, *_, drive, _ = episode(tmp_path, Script(moving=True), handback=BACK, expert=EXPERT)
    run(client, drive)
    out = flags_of(client, recorder)["interventions"][0]["max_delta_action"]["handback"]
    assert out == pytest.approx(0.5 / hb.JUMP_HZ, abs=0.005)                    # panda_joint1 moves 0.5 rad/s


def test_the_expert_is_not_stopped_without_a_handback(tmp_path):
    client, recorder, ctx, drive, _ = episode(tmp_path, TWO, expert=EXPERT)
    run(client, drive)
    assert ctx.stop() is False and recorder.takes[-1].end_t is None and not any(isinstance(x, tuple) and x[0] == "stopped" for x in LOG)


# ------------------------------------------------------------------ the spec
def test_the_handback_spec():
    assert hb.parse(None) is None and hb.parse("none") is None
    assert hb.parse({}) == hb.Handback("phase_done", 3.0, 0.3)
    assert hb.parse({"when": "after_s", "after_s": 2, "settle_s": 0}) == hb.Handback("after_s", 2.0, 0.0)
    for bad, match in [("phase_done", "expected `none` or a mapping"), ({"when": "never"}, "handback.when"), ({"x": 1}, "handback.x: unknown key"),
                       ({"when": "after_s", "after_s": 0}, "handback.after_s"), ({"settle_s": -1}, "handback.settle_s"),
                       ({"after_s": 2}, "only for `when: after_s`"), ({"settle_s": True}, "handback.settle_s")]:
        with pytest.raises(SpecError, match=match):
            hb.parse(bad)


def test_max_interventions_defaults_and_needs_a_handback():
    def made(**live):
        spec = spec_of(triggers=[PHASE_TIME], **live)
        return REGISTRY.create("live", "dagger", spec, spec.live.params)
    assert made().max_interventions == 1 and made(handback=BACK).max_interventions == 3 and made(handback=BACK, max_interventions=2).max_interventions == 2
    for live, match in [({"max_interventions": 2}, "needs a `handback`"), ({"handback": BACK, "max_interventions": 0}, "max_interventions"),
                        ({"handback": BACK, "max_interventions": "2"}, "max_interventions"), ({"handback": {"when": "x"}}, "handback.when")]:
        with pytest.raises(SpecError, match=match):
            made(**live)


def test_when_the_stop_condition_holds():
    phases = ["approach", "grasp", "open"]
    done, after = hb.Handback(), hb.Handback("after_s", 2.0)
    assert done.reason(phases, "approach", "approach", 9) is None and done.reason(phases, "grasp", "approach", 9) is None
    assert done.reason(phases, "approach", None, 9) is None and done.reason(phases, "approach", "open", 0) == "phase_done: approach -> open"
    assert after.reason(phases, "approach", "approach", 1.9) is None and after.reason(phases, "approach", "approach", 2.0) == "after_s: 2.0 s"


def test_jumps_measure_the_steps_of_the_window_only():
    t, q = [0.0], [{"a": 0.0, "b": 0.0}]
    j = hb.Jumps(lambda: q[0], ["a", "b"], lambda: t[0])
    assert not j.watching
    j.arm()
    for k, (dt, a) in enumerate([(0.0, 0.0), (0.01, 5.0), (0.04, 0.1), (0.04, 0.4), (2.0, 9.0)]):      # the 10 ms sample is skipped
        t[0] += dt
        q[0] = {"a": a, "b": 0.0}
        j.sample()
    assert j.peak == pytest.approx(0.3) and not j.watching
