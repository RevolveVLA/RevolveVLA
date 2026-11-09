"""No-sim checks of the spec bridge and the world copies (fast; run on the host)."""
import re

import pytest

import cli
import phases as ph
import spec
from revolve_common import specs

TASKS = ("open_fridge_door", "open_drawer")


@pytest.mark.parametrize("task", TASKS)
def test_legacy_thresholds_are_the_old_plugin_values(task):
    """episode.json thresholds for the offline verdict, from task.yaml, equal the old plugin.yaml values
    (task plugin replay: + simulator checks:)."""
    r = spec.task(task)
    th = spec.legacy_thresholds(r)
    key = "door_open_min_deg" if task == "open_fridge_door" else "drawer_open_min_cm"
    assert th == {key: 60.0 if task == "open_fridge_door" else 15.0, "hold_s": 1.0, "handle_slip_max_m": 0.01,
                  "spawn_clearance_min_m": 0.0, "peak_body_speed_max_mps": 5.0, "penetration_max_m": 0.01,
                  "grasp_width_tol_m": 0.006}
    lt = spec.legacy_task(r)["replay"]
    assert (lt["pull_slowdown"], lt["pull_min_s"], lt["budget_s"]) == (3.0, 8.0, 30.0)


def test_hold_and_half_width():
    f, d = spec.task("open_fridge_door"), spec.task("open_drawer")
    assert spec.hold(f)[:3] == ("handle", "door", "deg") and spec.hold(d)[:3] == ("bar", "drawer", "cm")
    assert spec.grasp_half_width(f, [0.012, 0.014, 0.25]) == 0.007           # size[1] / 2
    assert spec.grasp_half_width(d, [0.013, 0.624, 0.014]) == 0.007          # min(size[1], size[2]) / 2


@pytest.mark.parametrize("task", TASKS)
def test_task_phases_match_the_boundary_rules(task):
    assert tuple(p.name for p in spec.task(task).task.phases) == ph.PHASES


def test_sanity_violation_reads_the_condition():
    c = specs.Condition(metric="penetration_m", lt=0.01)
    assert not ph.violates(c, 0.009) and ph.violates(c, 0.01)
    c = specs.Condition(metric="other_robot_contacts", eq=0)
    assert ph.violates(c, 1) and not ph.violates(c, 0)


@pytest.mark.parametrize("world", ["kitchen_partnet.sdf", "kitchen_partnet_robot.sdf", "kitchen_partnet_skills.sdf"])
def test_world_copy_render_off(tmp_path, world):
    w = cli.world_copy(spec.WORLDS / world, tmp_path, video=False, module="phase_replay_system",
                       robot_uri="/x/mobile_panda", rtf=0)
    t = w.read_text()
    assert "scene_camera" not in t and "sensors-system" not in t and "recorder" not in t
    assert re.findall(r"<module_name>(\w+)</module_name>", t) == ["phase_replay_system"]
    assert "<real_time_factor>0</real_time_factor>" in t
    assert ("model://mobile_panda" not in t) and (world == "kitchen_partnet.sdf" or "/x/mobile_panda" in t)


def test_scene_spec_points_at_the_ported_world():
    r = spec.task("open_fridge_door")
    assert spec.path(r.scene.world).is_file() and spec.path(r.scene.world).parent == spec.WORLDS
    assert spec.path(r.robot.franka_description).joinpath("robots").is_dir()
