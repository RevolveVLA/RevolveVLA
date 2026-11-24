"""plugins/live/dagger.py: the handover (check before act, latched, the policy stops before the expert starts), the per-phase
quota, every trigger kind, the sidecar flags, the expert's failure."""
import fake_live as L
import pytest
from datagen.core.spec import SpecError
from evaluation import rollout_client as C
from evaluation.oracle_actions import ActionSeries
from fake_live import LOG, LiveClient, Script
import numpy as np
from recovery.core.interfaces import Gate, LiveContext
from recovery.core.registry import REGISTRY
from recovery.core.spec import parse
from recovery.plugins.live import dagger
from revolve_common.sim_fake import FakeSession

PHASE_TIME = {"plugin": "phase_time", "phase": "approach", "after_s": 1.5}
STALL = {"plugin": "stall", "phase": "approach", "window_s": 1.0, "min_motion": 0.01}
SLIP = {"plugin": "condition", "phase": "open", "value": "open.while[1]", "op": "gt", "threshold": 0.01, "for_s": 0.2}


def spec_of(**live):
    return parse({"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": "e.yaml"}, "out": "o",
                  "live": {"plugin": "dagger", "expert": {"plugin": "fake_takeover"}, **live}})


@pytest.fixture(autouse=True)
def stub_state(monkeypatch):
    LOG.clear()
    monkeypatch.setattr(dagger, "RosState", L.StateStub)


def episode(tmp_path, script=None, closed=(), **live):
    """(client, recorder, ctx, drive, published): a recorder wrapping the oracle-style stream of a scripted client."""
    live = {"triggers": [PHASE_TIME], **live}
    spec = spec_of(**live)
    recorder = REGISTRY.create("live", "dagger", spec, spec.live.params)
    session = FakeSession(tmp_path / "scene.yaml", "mobile_panda", "open_fridge_door", tmp_path / "sim", "n", 100)
    session.start()
    client = LiveClient(session, script or Script())
    session.passes = lambda n: client.finished
    ctx = LiveContext(client, spec.task, spec.robot, tmp_path / "scene.yaml", tmp_path / "sim", Gate(), frozenset(closed))
    ticks = 40 * 30
    series = ActionSeries({"arm": ["j"]}, {"arm": np.arange(ticks, dtype=float)[:, None]}, np.zeros((ticks, 3)))
    published = []

    def policy_drive(n):
        return {"steps": C.stream(client, n, series, lambda pos, base: (published.append(pos["arm"][0]), LOG.append("policy")),
                                  sleep=lambda dt: None, halt=ctx.gate)}
    return client, recorder, ctx, recorder.driver(policy_drive, ctx), published


def run(client, drive):
    n = client.start_episode()
    extra = drive(n)
    return extra, client.wait_files(client.wait_result(n) and n)


def test_the_expert_is_prepared_before_the_episode_starts_and_the_policy_stops_before_it_takes_over(tmp_path):
    client, recorder, ctx, drive, published = episode(tmp_path)
    assert LOG == [("prepare", "open_fridge_door")]
    extra, result = run(client, drive)
    assert LOG[1] == ("start", 0) and LOG[2] == "policy" and LOG[-1][0] == "takeover" and LOG[-1][1].startswith("phase_time:")
    assert LOG.count("policy") == len(published) == extra["steps"] and LOG.index(LOG[-1]) > LOG.index("policy")
    assert extra["expert"] == "fake_v1" and extra["notes"] == "ok" and result["passed"]
    assert 1.5 <= recorder.taken.t < 1.7 and published == sorted(published) and 40 <= len(published) <= 50    # 1.5 s at 30 Hz


def test_the_check_runs_before_each_action_so_the_takeover_tick_is_the_first_expert_action(tmp_path):
    client, recorder, ctx, drive, published = episode(tmp_path)
    run(client, drive)
    assert ctx.gate() is True                                      # latched: stays true, the policy is never asked again
    n_before = len(published)
    assert recorder.check() is True and len(published) == n_before
    assert LOG.count(("start", 0)) == 1 and sum(isinstance(x, tuple) and x[0] == "takeover" for x in LOG) == 1


def test_a_closed_phase_is_never_taken_over_the_verdict_stands(tmp_path):
    client, recorder, ctx, drive, published = episode(tmp_path, closed=["approach"])
    extra, result = run(client, drive)
    assert recorder.taken is None and not result["passed"] and "expert" not in extra and extra["steps"] == 600
    assert not any(isinstance(x, tuple) and x[0] == "takeover" for x in LOG)


def test_a_phase_that_is_open_still_gets_its_takeover_after_a_closed_one(tmp_path):
    script = Script(values={"open.while[1]": 0.05})
    client, recorder, ctx, drive, _ = episode(tmp_path, script, closed=["approach", "grasp"], triggers=[PHASE_TIME, SLIP])
    run(client, drive)
    assert (recorder.taken.phase, recorder.taken.trigger) == ("open", "condition") and 9.2 <= recorder.taken.t < 9.5


@pytest.mark.parametrize("trigger, script, phase, name", [
    (PHASE_TIME, Script(), "approach", "phase_time"),
    (STALL, Script(only_phase="approach"), "approach", "stall"),
    (SLIP, Script(values={"open.while[1]": 0.05}), "open", "condition")])
def test_every_trigger_kind_hands_over_to_the_expert_which_recovers(tmp_path, trigger, script, phase, name):
    client, recorder, ctx, drive, _ = episode(tmp_path, script, triggers=[trigger])
    extra, result = run(client, drive)
    assert (recorder.taken.phase, recorder.taken.trigger) == (phase, name) and result["passed"] and extra["expert"] == "fake_v1"


def test_a_moving_robot_is_not_stalled(tmp_path):
    client, recorder, ctx, drive, _ = episode(tmp_path, Script(end_s=4.0, moving=True, only_phase="approach"), triggers=[STALL])
    run(client, drive)
    assert recorder.taken is None


def test_the_sidecar_flags_of_a_takeover(tmp_path):
    client, recorder, ctx, drive, _ = episode(tmp_path)
    run(client, drive)
    flags = recorder.finish({"episode": 0, "result": client.wait_files(0)})
    assert set(flags) == {"takeover_t", "takeover_sim_t", "trigger", "reason", "phase", "max_delta_action", "expert",
                          "expert_failure", "segments", "segment"}
    assert (flags["trigger"], flags["phase"], flags["expert"], flags["max_delta_action"], flags["expert_failure"]) == (
        "phase_time", "approach", "fake_v1", 0.02, False)
    assert flags["takeover_t"] == pytest.approx(1.5, abs=0.2) and flags["takeover_sim_t"] == pytest.approx(L.START + flags["takeover_t"])
    assert flags["segments"] == [[0.0, flags["takeover_t"], "policy"], [flags["takeover_t"], 3.0, "expert"]] and flags["segment"] is None


def test_record_interventions_asks_for_the_expert_span_only(tmp_path):
    client, recorder, ctx, drive, _ = episode(tmp_path, record="interventions")
    run(client, drive)
    flags = recorder.finish({"episode": 0, "result": client.wait_files(0)})
    assert flags["segment"] == [flags["takeover_t"], 3.0] and flags["segments"][-1] == [*flags["segment"], "expert"]


def test_an_episode_without_a_takeover_is_one_policy_segment(tmp_path):
    client, recorder, ctx, drive, _ = episode(tmp_path, Script(end_s=3.0), triggers=[{**PHASE_TIME, "after_s": 10}])
    run(client, drive)
    flags = recorder.finish({"episode": 0, "result": client.wait_files(0)})
    assert flags["takeover_t"] is None and flags["segments"] == [[0.0, 20.0, "policy"]] and flags["expert_failure"] is False


def test_an_expert_that_fails_or_raises_is_flagged_not_fatal(tmp_path, capsys):
    client, recorder, ctx, drive, _ = episode(tmp_path, expert={"plugin": "fake_takeover", "fail": True})
    run(client, drive)
    assert recorder.finish({"episode": 0, "result": client.wait_files(0)})["expert_failure"] is True
    LOG.clear()
    client, recorder, ctx, drive, _ = episode(tmp_path, expert={"plugin": "fake_takeover", "raises": True})
    extra, _ = run(client, drive)
    assert recorder.error == "RuntimeError: planner died" and extra["notes"] == recorder.error
    assert recorder.finish({"episode": 0, "result": client.wait_files(0)})["expert_failure"] is True


def test_the_quota_closes_a_phase_once_its_takeovers_are_taken(tmp_path):
    spec = spec_of(triggers=[PHASE_TIME], quota={"approach": 2, "grasp": 1})
    recorder = REGISTRY.create("live", "dagger", spec, spec.live.params)
    assert recorder.closed({}) == {"open"}                                       # not listed once a quota is given: none wanted
    assert recorder.closed({"approach": 1, "grasp": 1}) == {"grasp", "open"}
    assert recorder.closed({"approach": 2, "grasp": 5, "open": 9}) == {"approach", "grasp", "open"}
    free = spec_of(triggers=[PHASE_TIME])
    assert REGISTRY.create("live", "dagger", free, free.live.params).closed({"approach": 99}) == frozenset()


def test_validate_names_the_yaml_path_of_what_is_wrong(tmp_path):
    for live, match in [({}, "live.triggers"), ({"triggers": [PHASE_TIME], "bogus": 1}, "live.bogus: unknown key"),
                        ({"triggers": [PHASE_TIME], "record": "half"}, "live.record"),
                        ({"triggers": [PHASE_TIME], "quota": {"approach": -1}}, "live.quota"),
                        ({"triggers": [PHASE_TIME], "expert": {"nope": 1}}, "live.expert")]:
        with pytest.raises(SpecError, match=match):
            spec = spec_of(**live)
            REGISTRY.create("live", "dagger", spec, spec.live.params)
    for live, match in [({"triggers": [PHASE_TIME], "quota": {"fly": 1}}, "live.quota: unknown phase"),
                        ({"triggers": [{"plugin": "phase_time", "phase": "fly", "after_s": 1}]}, "live.triggers\\[0\\].phases"),
                        ({"triggers": [{"plugin": "nope"}]}, "live.triggers\\[0\\]")]:
        spec = spec_of(**live)
        with pytest.raises(SpecError, match=match):
            REGISTRY.create("live", "dagger", spec, spec.live.params).validate(spec)
    spec = spec_of(triggers=[PHASE_TIME], expert={"plugin": "fake_takeover"})
    REGISTRY.create("live", "dagger", spec, spec.live.params).validate(spec)
