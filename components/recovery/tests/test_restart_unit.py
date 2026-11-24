"""core/restart.py: the restart time and the restart state read from the bag, the reproducibility gate."""
import fake_recovery as F
import pytest
from recovery.core.restart import Restart, restart_at, restart_time
from revolve_common.supervision import BagSource


def source(tmp_path, t: F.Try):
    run = F.write_eval_run(tmp_path, [t])
    rec = F.json.loads((run / "results.json").read_text())["sets"][t.set]["rollouts"][f"{t.seed}_r0"]["tries"][0]["record"]
    return BagSource(F.read_messages, rec["bag"], F.ROBOT, F.json.loads(F.Path(rec["result_file"]).read_text()))


@pytest.mark.parametrize("restart_from, back_s, t_fail, start, expected", [
    ("phase_start", 1.0, 11.0, 6.0, 6.0),
    ("before_failure", 1.0, 11.0, 6.0, 10.0),
    ("before_failure", 8.0, 11.0, 6.0, 6.0),          # never before the phase's start
    ("before_failure", 0.0, 11.0, 6.0, 11.0),
])
def test_restart_time(restart_from, back_s, t_fail, start, expected):
    assert restart_time(restart_from, back_s, t_fail, start) == expected


def test_state_at_the_restart(tmp_path):
    r = restart_at(source(tmp_path, F.Try(failed="grasp")), 4.0, F.ARM, 0.02)
    assert r.reproducible and r.t == 4.0 and r.reason is None
    assert r.arm == pytest.approx(tuple(0.1 * (i + 1) + 0.5 * 4 / 6 for i in range(7)), abs=1e-5)
    assert r.base == pytest.approx((-1.75, 1.1 + 0.8 * 4 / 6, 1.5), abs=1e-5) and r.object_joint == 0.0


def test_an_opened_door_is_unreproducible_with_its_value(tmp_path):
    r = restart_at(source(tmp_path, F.Try(failed="open")), 20.0, F.ARM, 0.02)      # the door opens 0.1 rad/s from t = 9
    assert not r.reproducible and r.object_joint == pytest.approx(1.1)
    assert "fridge::top_door_hinge = 1.100 is beyond 0.02" in r.reason
    assert restart_at(source(tmp_path / "b", F.Try(failed="open")), 9.0, F.ARM, 0.02).reproducible    # still closed at the phase start
    assert restart_at(source(tmp_path / "c", F.Try(failed="grasp", door=0.03)), 4.0, F.ARM, 0.05).reproducible    # a wider gate


def test_missing_robot_state_is_unreproducible(tmp_path):
    src = source(tmp_path, F.Try(failed="grasp"))
    r = restart_at(src, -5.0, F.ARM, 0.02)                                           # before the first message
    assert not r.reproducible and "no robot state in the bag at t=-5.00" in r.reason and r.base is None
    assert not restart_at(src, 4.0, [*F.ARM, "missing_joint"], 0.02).reproducible


def test_dict_round_trip():
    r = Restart(1.0, (1.0, 2.0, 3.0), (0.1, 0.2), 0.0)
    assert Restart.from_dict(r.to_dict()) == r
    assert Restart.from_dict(Restart(1.0, reason="x").to_dict()) == Restart(1.0, reason="x")
