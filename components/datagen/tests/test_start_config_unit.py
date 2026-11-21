"""PRD 08: the start design lives in task.yaml `starts:` and is read through specs_bridge.starts. The equivalence test
holds every (task, robot, variant) to data/start_config_before.json, written from the generator.yaml + planner.py
code before the move: the ROUND_STARTS text, the parsed spec the replay's handoff meta derives from (sequence,
slowdown), the box centre, the timing / pull blocks."""
import dataclasses
import json
from pathlib import Path

import pytest

import specs_bridge
import start_design
from revolve_common import specs

BEFORE = json.loads((Path(__file__).parent / "data" / "start_config_before.json").read_text())
# fridge on the SO-101s: generator.yaml gave them the panda's clustered block but their planner.py had no box centre,
# so plan.py stopped ("no box centre"). They have no start design now (whole-body IK) instead of a plan that fails.
WAS_UNPLANNABLE = {"open_fridge_door/mobile_so101/None", "open_fridge_door/mobile_so101_xl/None"}
BOX_LIMITS = {"x_m": 0.17, "y_m": 0.17, "yaw_deg": 27.0}


def after(task, robot, variant):
    try:
        design = specs_bridge.starts(task, robot, variant)
    except ValueError:
        return "error"
    blocks = specs_bridge.round_blocks(task)
    out = dict(round_starts=None, spec=None, centre=None, timing=json.dumps(blocks.get("timing")),
               pull=json.dumps(blocks.get("pull")))
    if design is not None:
        out.update(round_starts=json.dumps(design.block()),
                   spec=json.loads(json.dumps(dataclasses.asdict(start_design.parse_starts(design.block())))))
        if design.method == "box":
            out["centre"] = list(design.centre)
    return out


@pytest.mark.parametrize("key", sorted(BEFORE))
def test_start_config_is_identical_to_before(key):
    task, robot, variant = key.split("/")
    want, got = BEFORE[key], after(task, robot, None if variant == "None" else variant)
    if key in WAS_UNPLANNABLE:
        assert want["centre"] is None and want["round_starts"] is not None
        want = dict(want, round_starts=None, spec=None)
    elif want != "error" and want["spec"] is not None and "mode" in json.loads(want["round_starts"]):
        want = dict(want, centre=None)            # a reach / drive block never used the box centre
    assert got == want


def test_every_task_and_robot_is_covered():
    keys = {(t, r) for t, r, _ in (k.split("/") for k in BEFORE)}
    tasks = {p.name for p in (specs_bridge.SPECS / "tasks").iterdir()}
    robots = {p.name for p in (specs_bridge.SPECS / "robots").iterdir()}
    assert keys == {(t, r) for t in tasks for r in robots}


def test_box_hard_limits_are_those_of_before():
    for task in ("open_fridge_door", "open_drawer"):
        assert specs_bridge.starts(task, "mobile_panda").base_limits() == BOX_LIMITS


def test_accessor_defaults_to_the_environment(monkeypatch):
    monkeypatch.setenv(specs.ROBOT_ENV, "mobile_so101_xl")
    monkeypatch.delenv(specs.VARIANT_ENV, raising=False)
    assert specs_bridge.starts("open_drawer").method == "drive"
    monkeypatch.setenv(specs.ROBOT_ENV, "mobile_panda")
    monkeypatch.setenv(specs.VARIANT_ENV, "drive_seq")
    assert specs_bridge.starts("open_fridge_door").block()["sequence"] == "drive_then_arm"
    with pytest.raises(ValueError, match="datagen variant 'drive_seq'"):
        specs_bridge.starts("open_drawer")


def test_every_task_yaml_block_is_accepted_by_the_planner_parser():
    """common's schema and start_design.parse_starts (the container's trust boundary) agree on every shipped block."""
    for task in (specs_bridge.SPECS / "tasks").iterdir():
        t = specs.parse(specs.Task, task / "task.yaml")
        for robot, entry in t.starts.items():
            for d in [entry, *entry.variants.values()]:
                start_design.parse_starts(d.block())


def test_generator_yaml_has_no_starts():
    assert all("starts" not in t and "variants" not in t and "robots" not in t
               for t in specs_bridge.generator()["tasks"].values())
