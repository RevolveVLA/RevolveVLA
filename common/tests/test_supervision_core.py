"""Supervisor + build: priority, latch, start(), phase scoping, and the spec validation errors."""
import pytest

from conftest import SPECS
from revolve_common import specs
from revolve_common.supervision import Held, Supervisor, TaskInfo, Tick, build
from revolve_common.supervision import triggers as registry


@pytest.fixture(scope="module")
def task():
    return specs.resolve(SPECS, "open_fridge_door", assets=False)


class Always:
    def __init__(self, name, phases=None, when=0.0):
        self.name, self.phases, self.when, self.resets = name, phases, when, 0

    def reset(self):
        self.resets += 1

    def check(self, tick):
        return f"{self.name}@{tick.t}" if tick.t >= self.when else None


def test_held_debounce():
    h = Held(0.2)
    assert [h.update(t, True) for t in (0.0, 0.1, 0.2)] == [False, False, True]
    assert not h.update(0.3, False) and not h.update(0.4, True)
    assert h.update(0.6, True)
    assert Held(0).update(5.0, True)


def test_first_trigger_in_list_order_wins_and_history_counts_all():
    sup = Supervisor([Always("a", when=1.0), Always("b")], hz=10)
    assert sup.check(Tick(0.0, "approach")).trigger == "b"
    sup = Supervisor([Always("a"), Always("b")])
    f = sup.check(Tick(0.5, "approach"))
    assert (f.trigger, f.reason, f.phase, f.t) == ("a", "a@0.5", "approach", 0.5)
    assert dict(sup.history) == {"a": 1, "b": 1}


def test_latched_until_start():
    a = Always("a", when=1.0)
    sup = Supervisor([a])
    assert sup.check(Tick(0.5, "approach")) is None
    f = sup.check(Tick(1.0, "approach"))
    assert sup.check(Tick(2.0, "grasp")) is f and sup.firing is f
    sup.start()
    assert sup.firing is None and not sup.history and sup.check(Tick(0.2, "approach")) is None


def test_phase_scope_and_reset_on_phase_change():
    sup = Supervisor([Always("a", phases=("grasp",)), Always("b", phases=("open",))])
    assert sup.check(Tick(0.0, "approach")) is None
    assert sup.check(Tick(1.0, "grasp")).trigger == "a"
    sup.start()
    assert sup.check(Tick(2.0, "open")).trigger == "b"


def test_resets_triggers_on_phase_change_and_ignores_idle_and_old_ticks():
    a = Always("a", when=9.0)
    sup = Supervisor([a])
    n = a.resets
    sup.check(Tick(0.0, None))
    assert a.resets == n                            # no episode: nothing happens
    sup.check(Tick(1.0, "approach"))
    sup.check(Tick(2.0, "approach"))
    assert a.resets == n + 1
    sup.check(Tick(3.0, "grasp"))
    assert a.resets == n + 2
    sup.check(Tick(2.5, "open"))                    # out of order: dropped
    assert a.resets == n + 2


def test_scan_stops_at_the_firing_tick():
    sup = Supervisor([Always("a", when=2.0)])
    f = sup.scan(Tick(float(t), "approach") for t in range(5))
    assert f.t == 2.0
    assert Supervisor([Always("a", when=9.0)]).scan([Tick(0.0, "approach")]) is None


def test_task_info(task):
    info = TaskInfo.of(task)
    assert list(info.phases) == ["approach", "grasp", "open"]
    assert info.phases["approach"].timeout_s == 20.0
    assert {"approach.done[0]", "approach.done[2]", "open.while[1]", "grasp.done[0]"} <= info.values
    assert info.has_value("sanity.body_speed") and info.sanity == {"body_speed", "penetration", "robot_contacts",
                                                                   "finite_cmds", "gz_log"}
    assert TaskInfo.of(task.task).has_value("sanity.anything")      # a bare Task: sanity keys unknown
    ok = {"approach.done[0]": 0.01, "approach.done[1]": 0.1, "approach.done[2]": 0.1}
    assert info.is_done("approach", ok)
    assert not info.is_done("approach", {**ok, "approach.done[0]": 0.5})
    assert not info.is_done("approach", {**ok, "approach.done[1]": None})
    assert not info.is_done("global", ok)


def test_build_orders_and_maps_phase_shorthand(task):
    sup = build([{"plugin": "verdict"}, {"plugin": "phase_time", "phase": "open", "fraction": 0.5},
                 {"plugin": "stall", "phases": ["approach"], "window_s": 3, "min_motion": 0.01}], task, hz=20)
    assert [t.name for t in sup.triggers] == ["verdict", "phase_time", "stall"]
    assert [t.phases for t in sup.triggers] == [None, ("open",), ("approach",)]
    assert sup.period == pytest.approx(0.05)


@pytest.mark.parametrize("spec, msg", [
    ({"plugin": "verdict", "phases": ["lift"]}, "triggers[0].phases: unknown phase ['lift'] (phases: approach"),
    ({"plugin": "nope"}, "triggers[0].plugin: no trigger 'nope'"),
    ({"plugin": "condition", "value": "open.done[3]", "op": "gt", "threshold": 1}, "triggers[0].value: unknown value"),
    ({"plugin": "condition", "value": "open.done[0]", "op": "gt"}, "triggers[0].threshold: Field required"),
    ({"plugin": "condition", "value": "open.done[0]", "op": "~", "threshold": 1}, "triggers[0].op"),
    ({"plugin": "stall", "window_s": 1, "min_motion": 0.1, "speed": 3}, "triggers[0].speed: Extra inputs"),
    ({"plugin": "phase_time"}, "exactly one of fraction, after_s"),
    ({"plugin": "phase_time", "fraction": 0.5, "after_s": 3}, "exactly one of fraction, after_s"),
    ({"plugin": "phase_time", "fraction": -1}, "must be > 0"),
    ({"color": "red"}, "a trigger is a mapping with a `plugin` key"),
])
def test_build_errors_name_the_yaml_path(task, spec, msg):
    with pytest.raises(specs.SpecError) as e:
        build([spec], task)
    assert msg in str(e.value)


def test_error_path_follows_the_list_index_and_prefix(task):
    with pytest.raises(specs.SpecError, match=r"live.triggers\[1\].phases"):
        build([{"plugin": "verdict"}, {"plugin": "verdict", "phases": ["x"]}], task, path="live.triggers")


def test_phase_time_fraction_needs_a_timeout(task):
    t = specs.resolve(SPECS, "open_fridge_door", assets=False).task.model_copy(deep=True)
    t.phases[1].timeout_s = None
    with pytest.raises(specs.SpecError, match=r"triggers\[0\].fraction: phase \['grasp'\] has no timeout_s"):
        build([{"plugin": "phase_time", "phases": ["grasp"], "fraction": 0.5}], t)
    build([{"plugin": "phase_time", "fraction": 0.5}], t)       # all phases: the ones without a timeout never fire


def test_external_trigger_from_REVOLVE_TRIGGER_PATH(task, tmp_path, monkeypatch):
    (tmp_path / "my_trigger_x.py").write_text(
        "from revolve_common.supervision import TriggerSpec\n"
        "from revolve_common.supervision.triggers import register\n"
        "@register('my_trigger_x')\n"
        "class X:\n"
        "    Spec = TriggerSpec\n"
        "    def __init__(self, params, task, **_): self.phases = None\n"
        "    def reset(self): pass\n"
        "    def check(self, tick): return 'x'\n")
    monkeypatch.setenv(registry.PATH_ENV, str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert build([{"plugin": "my_trigger_x"}], task).check(Tick(0.0, "approach")).reason == "x"
    finally:
        registry.PLUGINS.pop("my_trigger_x", None)
