"""specs: the scene / robot / task / sanity schemas, the entity checks against model SDFs, the sanity merge.
Self-contained (a small spec tree + fake assets in tmp), plus the repo's own specs (names only without assets/)."""
import textwrap

import pytest
import yaml

from revolve_common import specs
from revolve_common.specs import SpecError
from conftest import REPO, SPECS

CABINET_SDF = """<?xml version="1.0"?>
<sdf version="1.11"><model name="cabinet">
  <link name="body"><collision name="body_c0"/><collision name="body_c1"/></link>
  <link name="drawer"><collision name="drawer_c0"/><collision name="handle_bar"/></link>
  <joint name="world_fixed" type="fixed"><parent>world</parent><child>body</child></joint>
  <joint name="slide" type="prismatic"><parent>body</parent><child>drawer</child>
    <axis><xyz>1 0 0</xyz><limit><lower>0</lower><upper>0.35</upper></limit></axis></joint>
</model></sdf>
"""

SANITY = """
body_speed: {metric: peak_body_speed, lt: 5.0}
robot_contacts: {metric: contacts, from: robot, to: any_except_targets, eq: 0}
gz_log: {metric: gz_log_problem_lines, eq: 0, timed: false, ignore: ['benign line']}
"""

TASK = """
name: open
scene: room
prompt: open the drawer
targets: {handle: cabinet::drawer::handle_bar}
budget_s: 30
phases:
  - name: approach
    timeout_s: 10
    done:
      - {metric: distance, from: robot::tcp, to: handle, le: 0.015}
      - {metric: angle, from: robot::tcp, to: handle, axes: [z, -x], le: 0.13}
  - name: grasp
    done: [{metric: contacts, from: robot::fingers, to: handle, eq: 2, for_s: 0.5}]
  - name: pull
    done: [{metric: joint, of: cabinet::slide, ge: 0.15, for_s: 1.0}]
    while:
      - {metric: contacts, from: robot::fingers, to: handle, eq: 2}
      - {metric: slip, from: robot::tcp, to: handle, le: 0.013}
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


@pytest.fixture
def tree(tmp_path):
    """tmp/specs/{scenes/room, robots/bot, tasks/open, tasks/sanity.yaml} + tmp/assets/{a,b}/<models>."""
    write(tmp_path / "assets/a/cabinet/model.config", "<model><name>cabinet</name><sdf version='1.11'>cab.sdf</sdf>"
                                                       "</model>")
    write(tmp_path / "assets/a/cabinet/cab.sdf", CABINET_SDF)
    write(tmp_path / "assets/b/box/model.sdf", "<sdf><model name='box'><link name='link'/></model></sdf>")
    s = tmp_path / "specs"
    write(s / "scenes/room/scene.yaml", """
        name: room
        assets: [assets/a, assets/b]
        models:
          cabinet: {model: cabinet, pose: [-2.6, 0.3, 0, 0], joints: {slide: 0.1}}
          box: {model: "model://box"}
        robot_start: {base: [0, 0, 0], arm: ready}
        """)
    write(s / "robots/bot/robot.urdf.xacro", "<robot name='bot'/>")
    write(s / "robots/bot/controllers.yaml", "{}")
    write(s / "robots/bot/robot.yaml", """
        name: bot
        description: robot.urdf.xacro
        controllers: controllers.yaml
        poses: {ready: [0, 0, 0], folded: [0, 1, 0]}
        frames: {tcp: tool, base: base_link}
        groups: {fingers: [finger_l, finger_r]}
        """)
    write(s / "tasks/sanity.yaml", SANITY)
    write(s / "tasks/open/task.yaml", TASK)
    return s


def edit(path, fn):
    d = yaml.safe_load(path.read_text())
    fn(d)
    path.write_text(yaml.safe_dump(d, sort_keys=False))


def task_with(tree, cond, where=("approach", "done")):
    """The tree's task with `cond` added to phase where[0]'s where[1] list."""
    p = tree / "tasks/open/task.yaml"
    edit(p, lambda d: next(ph for ph in d["phases"] if ph["name"] == where[0]).setdefault(where[1], []).append(cond))
    return p


# ------------------------------------------------------------------ the repo's specs
REPO_SPECS = specs.all_specs(SPECS)


def test_repo_specs_found():
    names = {p.relative_to(SPECS).as_posix() for p in REPO_SPECS}
    assert {"scenes/kitchen_partnet/scene.yaml", "tasks/sanity.yaml", "tasks/open_fridge_door/task.yaml",
            "tasks/open_drawer/task.yaml"} <= names


@pytest.mark.parametrize("path", REPO_SPECS, ids=lambda p: p.relative_to(SPECS).as_posix())
def test_repo_specs_load_without_assets(path):
    assert specs.load(path, assets=False) is not None


@pytest.mark.skipif(not (REPO / "assets" / "generated_partnet").is_dir(), reason="no assets/ (./revolve assets)")
@pytest.mark.parametrize("path", REPO_SPECS, ids=lambda p: p.relative_to(SPECS).as_posix())
def test_repo_specs_load_with_assets(path):
    assert specs.load(path) is not None


def test_repo_tasks_have_the_three_phases():
    for t in ("open_fridge_door", "open_drawer"):
        r = specs.resolve(SPECS, t, assets=False)
        assert [p.name for p in r.task.phases] == ["approach", "grasp", "open"]
        assert set(r.sanity) == {"body_speed", "penetration", "robot_contacts", "finite_cmds", "gz_log"}
        assert sum(p.timeout_s for p in r.task.phases) <= r.task.budget_s


# ------------------------------------------------------------------ valid inputs
def test_tree_loads_and_resolves(tree):
    scene = specs.load(tree / "scenes/room/scene.yaml")
    assert scene.models["box"].model == "box" and scene.models["box"].pose == [0, 0, 0, 0]
    r = specs.resolve(tree, "open", robot="bot")
    assert r.models["cabinet"].sdf.name == "cab.sdf" and r.models["cabinet"].links["drawer"] == {"drawer_c0",
                                                                                                 "handle_bar"}
    a = r.task.phases[0].done[1]
    assert (a.metric, a.op, a.value, a.params["axes"]) == ("angle", "le", 0.13, ("z", "-x"))
    g = r.task.phases[1].done[0]
    assert (g.op, g.value, g.for_s, g.entities()) == ("eq", 2, 0.5, {"from": "robot::fingers", "to": "handle"})
    assert [c.metric for c in r.task.phases[2].while_] == ["contacts", "slip"]
    assert r.sanity["body_speed"].timed and r.sanity["body_speed"].params == {"of": "robot"}   # the default
    assert not r.sanity["gz_log"].timed and r.sanity["gz_log"].params["ignore"] == ("benign line",)


def test_load_kinds(tree):
    assert isinstance(specs.load(tree / "robots/bot/robot.yaml"), specs.Robot)
    assert isinstance(specs.load(tree / "tasks/open/task.yaml"), specs.Task)
    assert set(specs.load(tree / "tasks/sanity.yaml")) == {"body_speed", "robot_contacts", "gz_log"}
    with pytest.raises(SpecError, match="not specs/"):
        specs.load(write(tree / "tasks/other.yaml", "a: 1"))


# ------------------------------------------------------------------ conditions
@pytest.mark.parametrize("cond, needle", [
    ({"metric": "distance", "from": "robot::tcp", "to": "handle", "le": 0.1, "hold_s": 1},
     r"metric distance: unknown key\(s\) \['hold_s'\]"),
    ({"metric": "distance", "from": "robot::tcp", "to": "handle", "lte": 0.1},
     r"exactly one of lt, le, eq, ge, gt is needed, got none \(unknown key\(s\) \['lte'\]\)"),
    ({"metric": "distance", "from": "robot::tcp", "to": "handle", "le": 0.1, "ge": 0.0},
     r"exactly one of lt, le, eq, ge, gt is needed, got \['le', 'ge'\]"),
    ({"metric": "distance", "from": "robot::tcp", "to": "handle", "le": True}, "le needs a number"),
    ({"metric": "distance", "from": "robot::tcp", "le": 0.1}, "missing parameter 'to'"),
    ({"metric": "tcp_error", "le": 0.1}, "unknown metric 'tcp_error'"),
    ({"metric": "angle", "from": "robot::tcp", "to": "handle", "axes": ["z", "w"], "le": 0.1}, "axes must be"),
    ({"metric": "distance", "from": "robot::tcp", "to": "handle", "le": 0.1, "for_s": 0}, "for_s"),
    ({"metric": "gz_log_problem_lines", "eq": 0, "ignore": ["("]}, "bad regex"),
    ({"metric": "contacts", "from": "robot", "to": "handle", "eq": 0, "timed": False}, r"unknown key\(s\) \['timed'\]"),
])
def test_bad_condition_is_rejected(tree, cond, needle):
    with pytest.raises(SpecError, match=needle) as e:
        specs.load(task_with(tree, cond))
    assert "phases.0.done" in str(e.value)        # the location in the file


def test_unknown_task_key_is_rejected(tree):
    p = tree / "tasks/open/task.yaml"
    edit(p, lambda d: d.update(start={"base": [0, 0, 0]}))
    with pytest.raises(SpecError, match="start: Extra inputs are not permitted"):
        specs.load(p)


@pytest.mark.parametrize("entity, needle", [
    ("cabinet:::drawer", "not one of robot, robot::<alias>"),
    ("a::b::c::d", "not one of"),
    ("robot::a::b", "robot entities are robot or robot::<alias>"),
    ("fridge::door", "'fridge' is not a scene model"),
    ("nothing", "'nothing' is not a target alias or scene model"),
    ("cabinet::lid", r"model cabinet \(.*cab.sdf\) has no link or joint 'lid'"),
    ("cabinet::lid::c0", "model cabinet has no link 'lid'"),
    ("cabinet::drawer::knob", r"link cabinet::drawer has no collision 'knob' \(collisions: drawer_c0, handle_bar\)"),
    ("cabinet::slide", "a joint, but a collision or frame or link needed"),
    ("cabinet", "a whole model is a link group"),
    ("any_except_targets", "only work as contacts `to`"),
])
def test_bad_entity_is_rejected(tree, entity, needle):
    with pytest.raises(SpecError, match=needle):
        specs.load(task_with(tree, {"metric": "distance", "from": "robot::tcp", "to": entity, "le": 0.1}))


def test_joint_metric_needs_a_joint(tree):
    with pytest.raises(SpecError, match=r"cabinet::drawer: a link, but a joint needed"):
        specs.load(task_with(tree, {"metric": "joint", "of": "cabinet::drawer", "ge": 0.1}))
    with pytest.raises(SpecError, match=r"model cabinet .* has no link or joint 'hinge'"):
        specs.load(task_with(tree, {"metric": "joint", "of": "cabinet::hinge", "ge": 0.1}))


def test_contacts_take_groups_and_any(tree):
    p = task_with(tree, {"metric": "contacts", "from": "robot", "to": "cabinet", "eq": 0})
    task_with(tree, {"metric": "contacts", "from": "robot::fingers", "to": "any", "le": 2})
    assert specs.resolve(tree, "open", robot="bot").task.phases[0].done[-1].params["to"] == "any"
    assert specs.load(p) is not None


def test_without_assets_only_names_and_syntax_are_checked(tree):
    p = task_with(tree, {"metric": "distance", "from": "robot::tcp", "to": "cabinet::lid", "le": 0.1})
    assert specs.load(p, assets=False) is not None
    for bad in ("fridge::door", "cabinet:::x"):
        p = task_with(tree, {"metric": "distance", "from": "robot::tcp", "to": bad, "le": 0.1})
        with pytest.raises(SpecError):
            specs.load(p, assets=False)


def test_targets(tree):
    p = tree / "tasks/open/task.yaml"
    edit(p, lambda d: d["targets"].update(box="box::link"))
    with pytest.raises(SpecError, match="targets.box: also a scene model name"):
        specs.load(p)
    edit(p, lambda d: d.update(targets={"handle": "robot::tcp"}))
    with pytest.raises(SpecError, match="targets.handle: 'robot::tcp' is not a scene entity"):
        specs.load(p)
    edit(p, lambda d: d.update(targets={"handle": "cabinet::drawer::knob"}))
    with pytest.raises(SpecError, match="targets.handle: .*has no collision 'knob'"):
        specs.load(p)


@pytest.mark.parametrize("change, needle", [
    (lambda d: d["phases"].append(dict(d["phases"][0])), "duplicate names"),
    (lambda d: d["phases"][0].update(name="global"), "'global' is reserved"),
    (lambda d: d["phases"][1].update(done=[]), "phases.1.done: List should have at least 1 item"),
    (lambda d: d.update(budget_s=0), "budget_s"),
    (lambda d: d.update(name="shut"), "name 'shut' != its folder 'open'"),
    (lambda d: d.update(scene="garage"), "scene 'garage': no"),
])
def test_bad_task_is_rejected(tree, change, needle):
    p = tree / "tasks/open/task.yaml"
    edit(p, change)
    with pytest.raises(SpecError, match=needle):
        specs.load(p)


# ------------------------------------------------------------------ sanity
def test_sanity_base_holds_only_checks(tree):
    write(tree / "tasks/sanity.yaml", SANITY + "name: sanity\nphases: []\n")
    with pytest.raises(SpecError, match=r"(?s)name: not a check.*phases: not a check.*holds only checks"):
        specs.load(tree / "tasks/sanity.yaml")
    write(tree / "tasks/sanity.yaml", SANITY + "speed: {metric: peak_body_speed, lt: 5, every_s: 1}\n")
    with pytest.raises(SpecError, match=r"speed: metric peak_body_speed: unknown key\(s\) \['every_s'\]"):
        specs.resolve(tree, "open")


def test_sanity_merge_override_add_remove(tree):
    p = tree / "tasks/open/task.yaml"
    edit(p, lambda d: d.update(sanity={"body_speed": {"metric": "peak_body_speed", "lt": 2.0},
                                       "gz_log": None,
                                       "penetration": {"metric": "penetration", "lt": 0.005, "timed": False}}))
    r = specs.resolve(tree, "open")
    assert list(r.sanity) == ["body_speed", "robot_contacts", "penetration"]
    assert r.sanity["body_speed"].value == 2.0 and not r.sanity["penetration"].timed
    edit(p, lambda d: d.update(sanity={"nothing": None}))
    with pytest.raises(SpecError, match="sanity.nothing: null removes a base check, but tasks/sanity.yaml has no"):
        specs.resolve(tree, "open")


def test_sanity_entities_are_checked_per_task(tree):
    p = tree / "tasks/open/task.yaml"
    edit(p, lambda d: d.update(sanity={"lid": {"metric": "contacts", "from": "robot", "to": "cabinet::lid",
                                               "eq": 0}}))
    with pytest.raises(SpecError, match="sanity.lid.to: cabinet::lid: model cabinet"):
        specs.load(p)


def test_merge_function():
    a = specs.SanityCheck.model_validate({"metric": "penetration", "lt": 0.01})
    b = specs.SanityCheck.model_validate({"metric": "penetration", "lt": 0.02})
    assert specs.merge_sanity({"p": a, "q": a}, {"p": b, "q": None, "r": a}) == {"p": b, "r": a}


# ------------------------------------------------------------------ scene
@pytest.mark.parametrize("change, needle", [
    (lambda d: d["models"]["cabinet"].update(model="fridge"),
     r"models.cabinet.model: model://fridge is in none of \['assets/a', 'assets/b'\]"),
    (lambda d: d["models"]["cabinet"]["joints"].update(hinge=0.1), "models.cabinet.joints.hinge: .* has no joint"),
    (lambda d: d["models"]["cabinet"]["joints"].update(world_fixed=0.0), "a fixed joint"),
    (lambda d: d["models"]["cabinet"]["joints"].update(slide=0.5), r"outside the joint limits \[0.0, 0.35\]"),
    (lambda d: d["models"]["cabinet"].update(pose=[0, 0, 0]), "pose: List should have at least 4 items"),
    (lambda d: d["models"]["cabinet"].update(scale=2), "scale: Extra inputs"),
    (lambda d: d["models"].update(robot={"model": "box"}), "'robot': names are"),
    (lambda d: d.update(world="kitchen.sdf"), "world: Extra inputs"),
    (lambda d: d["robot_start"].update(base=[0, 0]), "robot_start.base"),
])
def test_bad_scene_is_rejected(tree, change, needle):
    p = tree / "scenes/room/scene.yaml"
    edit(p, change)
    with pytest.raises(SpecError, match=needle):
        specs.load(p)
    if "model://" in needle:            # names only: nothing to resolve
        assert specs.load(p, assets=False) is not None


# ------------------------------------------------------------------ robot, and a task with a robot
@pytest.mark.parametrize("change, needle", [
    (lambda d: d.update(description="missing.urdf.xacro"), "description: no file"),
    (lambda d: d["poses"].update(ready=[0, 0]), "every pose needs the same, nonzero number of joints"),
    (lambda d: d["groups"].update(tcp=["tool"]), r"\['tcp'\]: both a frame and a group"),
    (lambda d: d["groups"].update(palm=[]), "groups.palm: no links"),
    (lambda d: d.update(cameras=[]), "cameras: Extra inputs"),
])
def test_bad_robot_is_rejected(tree, change, needle):
    p = tree / "robots/bot/robot.yaml"
    edit(p, change)
    with pytest.raises(SpecError, match=needle):
        specs.load(p)


def test_robot_aliases_are_checked_with_a_robot(tree):
    p = task_with(tree, {"metric": "distance", "from": "robot::wrist", "to": "handle", "le": 0.1})
    assert specs.load(p) is not None                 # robot-free: checked when the task meets a robot
    with pytest.raises(SpecError, match=r"robot::wrist: robot bot has no frame or group 'wrist' \(frames: tcp, base"):
        specs.resolve(tree, "open", robot="bot")
    write(tree / "tasks/open/task.yaml", TASK)
    task_with(tree, {"metric": "distance", "from": "robot::fingers", "to": "handle", "le": 0.1})
    with pytest.raises(SpecError, match="robot::fingers: robot group 'fingers', but a collision or frame or link"):
        specs.resolve(tree, "open", robot="bot")


def test_robot_start_arm_is_checked_with_a_robot(tree):
    edit(tree / "scenes/room/scene.yaml", lambda d: d["robot_start"].update(arm="home"))
    assert specs.resolve(tree, "open").robot is None
    with pytest.raises(SpecError, match="robot_start.arm 'home' is not a pose of robot bot"):
        specs.resolve(tree, "open", robot="bot")
    edit(tree / "scenes/room/scene.yaml", lambda d: d["robot_start"].update(arm=[0, 0]))
    with pytest.raises(SpecError, match="robot_start.arm has 2 joints, robot bot 3"):
        specs.resolve(tree, "open", robot="bot")
    edit(tree / "scenes/room/scene.yaml", lambda d: d["robot_start"].update(arm=[0, 0.5, 0]))
    assert specs.resolve(tree, "open", robot="bot").robot.arm_dof == 3


def test_metric_registry_names():
    assert set(specs.METRICS) == {"joint", "distance", "angle", "contacts", "gripper_opening", "slip",
                                  "peak_body_speed", "penetration", "non_finite_commands", "gz_log_problem_lines"}
    assert all(p.type in ("entity", "axes", "patterns") for m in specs.METRICS.values() for p in m.params.values())


def test_scene_instance_outside_specs(tree, tmp_path):
    """A scene.yaml outside a specs/ tree (a randomizer's per-seed instance): validated like a scene, named like its
    base scene, assets relative to the repo of `root`."""
    d = yaml.safe_load((tree / "scenes/room/scene.yaml").read_text())
    d["robot_start"]["base"] = [0.5, -0.2, 1.0]
    p = write(tmp_path / "runs/exp/scene_7/scene.yaml", yaml.safe_dump(d))
    scene = specs.load(p, root=tree)
    assert scene.name == "room" and scene.robot_start.base == [0.5, -0.2, 1.0]
    d["models"]["cabinet"]["model"] = "nope"
    write(p, yaml.safe_dump(d))
    with pytest.raises(SpecError, match="model://nope"):
        specs.load(p, root=tree)
    with pytest.raises(SpecError, match="not under a specs/"):
        specs.load(write(tmp_path / "runs/exp/task.yaml", TASK))
