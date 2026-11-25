"""dataset.contract (DATASET-v2.md section 5): load, topics() and features() on the three real robot contracts and on
synthetic ones (x- anchors, duplicates, other sections, multi-source keys). topics() drives the host's bag check, so
every stream the porter reads, observations and actions alike, must be among them."""
import pytest
import yaml

from conftest import contract_action_topics, contract_path
from dataset import contract as C

CAMS = ["/wrist_cam/image/compressed", "/overhead_left_cam/image/compressed", "/overhead_right_cam/image/compressed"]
IMAGE_KEYS = ["observation.images.wrist", "observation.images.overhead_left", "observation.images.overhead_right"]
TOPICS = {
    "mobile_panda": [*CAMS, "/omni_wheel_drive_controller/odom", "/joint_states",
                     "/omni_wheel_drive_controller/cmd_vel", "/panda_arm_controller/joint_trajectory",
                     "/panda_hand_controller/joint_trajectory", "/task/prompt"],
    "mobile_so101": [*CAMS, "/omni_wheel_drive_controller/odom", "/joint_states",
                     "/omni_wheel_drive_controller/cmd_vel", "/arm_controller/joint_trajectory",
                     "/gripper_controller/joint_trajectory", "/task/prompt"],
}
TOPICS["mobile_so101_xl"] = TOPICS["mobile_so101"]
WIDTH = {"mobile_panda": 11, "mobile_so101": 9, "mobile_so101_xl": 9}


def test_load_is_the_yaml():
    # contract.load(path) -> the YAML document
    p = contract_path("mobile_panda")
    assert C.load(p) == yaml.safe_load(p.read_text())
    assert C.load(str(p)) == yaml.safe_load(p.read_text())


@pytest.mark.parametrize("robot", list(TOPICS))
def test_topics_of_real_contracts(robot):
    # contract.topics: every channel topic (observations, actions, tasks), file order, no duplicates;
    # DATASET-v2 5.3 DS3/DS7: both joint_trajectory topics, cmd_vel and /task/prompt are contract topics
    got = C.topics(C.load(contract_path(robot)))
    assert list(got) == TOPICS[robot]
    assert len(set(got)) == len(got)
    assert "/task/prompt" in got and "/omni_wheel_drive_controller/cmd_vel" in got
    assert sum(t.endswith("/joint_trajectory") for t in got) == 2
    assert not any(t.endswith("/controller_state") for t in got)        # DATASET-v2 12: diagnostic only


@pytest.mark.parametrize("robot", list(TOPICS))
def test_every_action_topic_is_checked(robot):
    # a bag without an action stream is refused up front (the porter would zero-fill that action for the whole
    # episode): the gripper, arm and base command topics are contract topics, in the actions section's order
    actions = contract_action_topics(robot)
    assert len(actions) == 3 and sum(t.endswith("/joint_trajectory") for t in actions) == 2, actions
    got = list(C.topics(C.load(contract_path(robot))))
    assert [t for t in got if t in actions] == actions


@pytest.mark.parametrize("robot", list(TOPICS))
def test_features_of_real_contracts(robot):
    # contract.features: 3 images (480, 480, 3) from resize; state / action widths 11/11 (panda), 9/9 (SO-101)
    got = C.features(C.load(contract_path(robot)))
    want = {**{k: (480, 480, 3) for k in IMAGE_KEYS}, "observation.state": (WIDTH[robot],),
            "action": (WIDTH[robot],)}
    assert {k: tuple(v) for k, v in got.items()} == want
    assert "task" not in got                                            # the tasks section is no feature


SYNTH = """
robot_type: synth
robot_interface: ros2
fps: 30
x-image: &image
  type: sensor_msgs/msg/CompressedImage
  qos: {reliability: best_effort}
x-hidden:
  channel: {topic: /hidden_in_anchor, type: std_msgs/msg/String}
observations:
  observation.images.front:
    channel: {topic: /front/image/compressed, <<: *image}
    apply: [resize: [240, 320]]
  observation.state:
    - channel: {topic: /odom, type: nav_msgs/msg/Odometry}
      select: [twist.twist.linear.x, twist.twist.angular.z]
    - channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
      select: [position.a, position.b, position.c]
  observation.effort:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    select: [effort.a]
actions:
  action:
    - channel: {topic: /cmd_vel, type: geometry_msgs/msg/TwistStamped}
      select: [twist.linear.x]
    - channel: {topic: /arm/joint_trajectory, type: trajectory_msgs/msg/JointTrajectory}
      select: [position.a, position.b, position.c]
rewards:
  next.reward:
    channel: {topic: /reward, type: std_msgs/msg/Float64}
    select: [data]
tasks:
  task:
    channel: {topic: /task/prompt, type: std_msgs/msg/String}
"""


def test_topics_synthetic_order_dedup_anchor_and_other_sections():
    # contract.topics: file order, a topic used twice listed once, multi-source keys contribute each source,
    # any other section (rewards) walked, x- anchor holders NOT walked
    got = C.topics(yaml.safe_load(SYNTH))
    assert list(got) == ["/front/image/compressed", "/odom", "/joint_states", "/cmd_vel", "/arm/joint_trajectory",
                         "/reward", "/task/prompt"]
    assert "/hidden_in_anchor" not in got


def test_features_synthetic():
    # contract.features: image (h, w, 3) from resize [h, w]; vectors = sum of the sources' select lengths;
    # rewards included; tasks not
    got = {k: tuple(v) for k, v in C.features(yaml.safe_load(SYNTH)).items()}
    assert got == {"observation.images.front": (240, 320, 3), "observation.state": (5,), "observation.effort": (1,),
                   "action": (4,), "next.reward": (1,)}


def test_topics_no_sections():
    # contract.topics: a contract with no channel sections has no topics (nothing invented)
    assert list(C.topics({"robot_type": "x", "fps": 30})) == []
