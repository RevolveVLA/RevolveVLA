"""The robots' Rosetta contracts (specs/robots/<r>/contract.yaml) and specs.check_contract, without Rosetta (the
simulator suite loads them with Rosetta's own loader). Each case edits a copy of mobile_panda's robot folder."""
import shutil

import pytest
import yaml

from revolve_common import specs
from conftest import REPO, SPECS

ROBOTS = sorted(p.parent.name for p in (SPECS / "robots").glob("*/robot.yaml"))
RECORD = REPO / specs.RECORD_YAML
CAMS = ["wrist_cam", "overhead_left_cam", "overhead_right_cam"]


def robot(name):
    return specs.load(SPECS / "robots" / name / "robot.yaml", assets=False)


@pytest.mark.parametrize("name", ROBOTS)
def test_repo_contracts_are_valid(name):
    assert specs.check_contract(robot(name), SPECS / "robots" / name, cameras=CAMS) == []
    assert specs.check_contract(robot(name), SPECS / "robots" / name, cameras=None) == []   # no xacro: any camera


def test_every_robot_has_a_contract():
    assert ROBOTS == ["mobile_panda", "mobile_so101", "mobile_so101_xl"]


@pytest.mark.parametrize("name, width", [("mobile_panda", 11), ("mobile_so101", 9), ("mobile_so101_xl", 9)])
def test_state_and_action_widths(name, width):
    c = yaml.safe_load((SPECS / "robots" / name / specs.CONTRACT_FILE).read_text())
    for sec, key in (("observations", "observation.state"), ("actions", "action")):
        assert sum(len(s["select"]) for s in c[sec][key]) == width, (sec, key)


@pytest.fixture
def panda(tmp_path):
    """A copy of mobile_panda's folder -> (robot, folder, edit(fn) on its contract as a dict)."""
    folder = tmp_path / "mobile_panda"
    shutil.copytree(SPECS / "robots" / "mobile_panda", folder, symlinks=True)

    def edit(fn):
        path = folder / specs.CONTRACT_FILE
        data = yaml.safe_load(path.read_text())
        fn(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    return robot("mobile_panda"), folder, edit


def errors(panda, cameras=CAMS):
    r, folder, _ = panda
    return specs.check_contract(r, folder, record=RECORD, cameras=cameras)


def test_copy_is_valid(panda):
    assert errors(panda) == []


def test_missing_contract(panda):
    (panda[1] / specs.CONTRACT_FILE).unlink()
    assert errors(panda) == [f"contract: no file {panda[1] / specs.CONTRACT_FILE}"]


def test_bad_yaml_and_duplicate_keys(panda):
    path = panda[1] / specs.CONTRACT_FILE
    path.write_text("fps: [30\n")
    assert errors(panda)[0].startswith("contract: ")
    path.write_text("fps: 30\nfps: 30\n")
    assert "duplicate key" in errors(panda)[0]


def state(d):
    return d["observations"]["observation.state"]


def action(d):
    return d["actions"]["action"]


@pytest.mark.parametrize("change, needle", [
    (lambda d: d.update(fps=15), "fps 15"),
    (lambda d: d.update(robot_type="mobile_so101"), "robot_type 'mobile_so101'"),
    (lambda d: d["observations"].pop("observation.images.wrist"), "image keys"),
    (lambda d: d["observations"].update({"observation.images.front": d["observations"]["observation.images.wrist"]}),
     "image keys"),
    (lambda d: d.pop("tasks"), "no tasks channel on /task/prompt"),
    (lambda d: d["tasks"]["task"]["channel"].update(topic="/prompt"), "no tasks channel"),
    (lambda d: d["observations"]["observation.images.wrist"]["channel"].update(topic="/wrist_cam/image"),
     "topic /wrist_cam/image is not recorded"),
    (lambda d: d["observations"]["observation.images.wrist"]["channel"].update(topic="/front_cam/image/compressed"),
     "topic /front_cam/image/compressed is not recorded"),
    (lambda d: action(d)[0]["channel"].update(topic="/cmd_vel"), "topic /cmd_vel is not recorded"),
    (lambda d: action(d)[1]["channel"].update(topic="/arm_controller/joint_trajectory"), "not recorded"),
    (lambda d: state(d)[1]["select"].append("position.panda_joint8"), "'panda_joint8' is not a joint"),
    (lambda d: action(d)[1]["select"].reverse(), "controller panda_arm_controller has"),
    (lambda d: action(d)[1]["select"].pop(), "controller panda_arm_controller has"),
    (lambda d: action(d).pop(2), "no action on the joint_trajectory of ['panda_hand_controller']"),
    (lambda d: state(d)[1]["select"].pop(), "state joints"),
    (lambda d: state(d).pop(0), None),     # the base twist is optional for these checks; Rosetta checks the rest
    (lambda d: action(d)[1].pop("channel"), "malformed"),
])
def test_bad_contract_is_rejected(panda, change, needle):
    panda[2](change)
    errs = errors(panda)
    if needle is None:
        assert errs == []
    else:
        assert any(needle in e for e in errs), errs


def test_cameras_come_from_the_robot(panda):
    assert any("not recorded" in e for e in errors(panda, cameras=["wrist_cam", "front_cam"]))


def test_record_only_topics_of_the_scene_are_allowed(panda):
    """A record-only `info` source on a scene topic (ground truth): recorded, so accepted here."""
    panda[2](lambda d: d.update(info={"info.door": {
        "channel": {"topic": "/world/kitchen_partnet/model/fridge/joint_state", "type": "sensor_msgs/msg/JointState",
                    "dtype": "float64"},
        "align": {"strategy": "hold", "timeline": "header"}, "select": ["position.top_door_hinge"]}}))
    assert errors(panda) == []


def test_validate_cli_checks_the_contract(tmp_path):
    """./revolve validate on the repo: every robot with its contract."""
    import subprocess
    r = subprocess.run([str(REPO / "revolve"), "validate", "--no-assets"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert sum("contract contract.yaml" in ln for ln in r.stdout.splitlines()) == len(ROBOTS)
