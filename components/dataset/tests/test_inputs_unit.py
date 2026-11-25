"""dataset.manifest.check_inputs (DATASET-v2.md sections 3 rule 4, 5.4, 6 "Fail closed", 8.3 last bullet, and the
decision "every topic the contract names, a 0-message topic counts as missing"): the result file and the bag's
metadata.yaml per passed episode, never the bag's messages."""
import json
from pathlib import Path

import pytest

from conftest import DROP, contract_topics, write_bag_metadata
from dataset import manifest as M

ROBOT, TASK = "mobile_panda", "open_fridge_door"
TOPICS = contract_topics(ROBOT)


def episode(p, entry, where=None):
    """The manifest.Episode for a Producer entry (built directly: check_inputs is tested on its own)."""
    return M.Episode(where=where or f"{p.root / 'manifest.json'} entry {p.entries.index(entry)}",
                     bag=p.root / entry["bag"], result=p.root / entry["result"], labels={}, duration_s=None)


def check(p, *entries, topics=TOPICS, task=TASK, robot=ROBOT):
    eps = [episode(p, e) for e in entries]
    return eps, M.check_inputs(eps, topics, task, robot)


def one_problem(probs, ep, *needles):
    """At least one `inputs:` problem, all of them about this episode, one naming every needle."""
    assert probs, "expected an inputs: problem"
    for x in probs:
        assert isinstance(x, str) and x.startswith("inputs:"), x
        assert ep.where in x, (ep.where, x)                     # names the manifest and the entry index
    assert any(all(n in x for n in needles) for x in probs), (needles, probs)


# ------------------------------------------------------------------ clean

def test_clean_episodes_and_duration_filled(producer):
    # check_inputs: no problem for complete inputs; sets episode.duration_s from the result
    p = producer()
    a = p.episode(duration_s=20.004)
    b = p.episode(duration_s=4.1)
    eps, probs = check(p, a, b)
    assert probs == []
    assert [e.duration_s for e in eps] == [20.004, 4.1]


def test_integer_and_zero_duration_ok(producer):
    # check_inputs: duration_s numeric non-negative (an int and 0 are numeric and non-negative)
    p = producer()
    a = p.episode(duration_s=20)
    b = p.episode(duration_s=0)
    eps, probs = check(p, a, b)
    assert probs == []
    assert eps[0].duration_s == 20 and eps[1].duration_s == 0


def test_result_without_task_or_robot_ok(producer):
    # check_inputs: the result's task / robot are compared only when present
    p = producer()
    e = p.episode(result_extra={"task": DROP, "robot": DROP})
    _, probs = check(p, e)
    assert probs == []


def test_extra_topics_and_controller_state_ignored(producer):
    # DATASET-v2 10.2: a bag holds many non-contract topics; they do not matter
    p = producer()
    e = p.episode(extra_topics=("/panda_arm_controller/controller_state", "/evaluator/state"))
    _, probs = check(p, e)
    assert probs == []


def test_never_reads_bag_messages(producer):
    # check_inputs: never reads bag messages (a bag of only metadata.yaml, no .mcap, passes)
    p = producer()
    e = p.episode()
    assert [f.name for f in (p.root / e["bag"]).iterdir()] == ["metadata.yaml"]
    assert check(p, e)[1] == []


def test_inputs_not_modified(producer):
    # DATASET-v2 3 rule 5: never modify inputs (results, bags)
    p = producer()
    e = p.episode()
    files = [p.root / e["result"], p.root / e["bag"] / "metadata.yaml"]
    before = [f.read_bytes() for f in files]
    check(p, e)
    assert [f.read_bytes() for f in files] == before


# ------------------------------------------------------------------ the result file

def test_missing_result(producer):
    # DATASET-v2 6 fail closed: a passed entry whose result file is missing
    p = producer()
    e = p.episode(result=False)
    eps, probs = check(p, e)
    one_problem(probs, eps[0])


def test_result_not_json(producer):
    # check_inputs: the result must be a JSON object
    p = producer()
    e = p.episode(result_text="{oops")
    eps, probs = check(p, e)
    one_problem(probs, eps[0])


@pytest.mark.parametrize("doc", [[], [{"passed": True, "duration_s": 1.0}], "passed", None])
def test_result_not_an_object(producer, doc):
    # check_inputs: the result must be a JSON object
    p = producer()
    e = p.episode(result_text=json.dumps(doc))
    eps, probs = check(p, e)
    one_problem(probs, eps[0])


@pytest.mark.parametrize("passed", [False, "true", 1, None, DROP], ids=["false", "str", "one", "null", "missing"])
def test_result_not_passed(producer, passed):
    # DATASET-v2 6/8.3: the result must say passed: true too (manifest said passed)
    p = producer()
    e = p.episode(result_extra={"passed": passed})
    eps, probs = check(p, e)
    one_problem(probs, eps[0])


@pytest.mark.parametrize("dur", [DROP, None, True, False, "20.0", -0.001, -5, [20.0], {"s": 1}],
                         ids=["missing", "null", "true", "false", "str", "neg-small", "neg-int", "list", "dict"])
def test_bad_duration(producer, dur):
    # check_inputs: a numeric non-negative duration_s, not bool
    p = producer()
    e = p.episode(result_extra={"duration_s": dur})
    eps, probs = check(p, e)
    one_problem(probs, eps[0])


@pytest.mark.parametrize("dur", [float("nan"), float("inf")], ids=["nan", "infinity"])
def test_non_finite_duration(producer, dur):
    # check_inputs: a NaN or Infinity duration_s (valid to Python's json) is an inputs: problem naming duration_s
    p = producer()
    e = p.episode(result_extra={"duration_s": dur})
    assert ("NaN" if dur != dur else "Infinity") in (p.root / e["result"]).read_text()
    eps, probs = check(p, e)
    one_problem(probs, eps[0], "duration_s")


@pytest.mark.parametrize("key, value", [("task", "open_drawer"), ("robot", "mobile_so101"), ("task", "")])
def test_result_task_robot_mismatch(producer, key, value):
    # check_inputs: the result's task / robot (when present) equal the given ones
    p = producer()
    e = p.episode(result_extra={key: value})
    eps, probs = check(p, e)
    one_problem(probs, eps[0])


# ------------------------------------------------------------------ the bag

def test_missing_bag_dir(producer):
    # DATASET-v2 6 fail closed: a missing bag directory
    p = producer()
    e = p.episode(bag=False)
    eps, probs = check(p, e)
    one_problem(probs, eps[0])


def test_bag_without_metadata(producer):
    # DATASET-v2 6 fail closed: a bag directory without metadata.yaml
    p = producer()
    e = p.episode(bag="empty")
    eps, probs = check(p, e)
    one_problem(probs, eps[0])


def test_bag_metadata_not_rosbag2(producer):
    # bag metadata: a metadata.yaml without rosbag2_bagfile_information / topics -> cannot prove the topics
    p = producer()
    e = p.episode()
    (p.root / e["bag"] / "metadata.yaml").write_text("something: else\n")
    eps, probs = check(p, e)
    one_problem(probs, eps[0])


@pytest.mark.parametrize("topic", TOPICS)
def test_each_contract_topic_absent(producer, topic):
    # decision: EVERY contract topic (observations, actions, /task/prompt) must be in the bag
    p = producer()
    e = p.episode(drop=(topic,))
    eps, probs = check(p, e)
    one_problem(probs, eps[0], topic)


@pytest.mark.parametrize("topic", ["/panda_hand_controller/joint_trajectory", "/task/prompt",
                                   "/wrist_cam/image/compressed", "/omni_wheel_drive_controller/cmd_vel"])
def test_topic_with_zero_messages(producer, topic):
    # decision / DATASET-v2 8.3: a topic with message_count 0 counts as missing
    p = producer()
    e = p.episode(zero=(topic,))
    eps, probs = check(p, e)
    one_problem(probs, eps[0], topic)


def test_follow_joint_trajectory_bag(producer):
    # DATASET-v2 5.4: an expert that used FollowJointTrajectory leaves no /<jtc>/joint_trajectory (only the
    # controller_state diagnostic): the passed entry fails; never filled from controller_state
    p = producer()
    jt = [t for t in TOPICS if t.endswith("/joint_trajectory")]
    e = p.episode(drop=jt, extra_topics=("/panda_arm_controller/controller_state",
                                         "/panda_hand_controller/controller_state"))
    eps, probs = check(p, e)
    one_problem(probs, eps[0], jt[0])
    one_problem(probs, eps[0], jt[1])


def test_so101_contract_topics(producer):
    # decision: the topics are the contract's (here mobile_so101's controllers), not hard-coded panda ones
    p = producer(robot="mobile_so101", task="open_drawer")
    e = p.episode(drop=("/gripper_controller/joint_trajectory",))
    eps = [episode(p, e)]
    probs = M.check_inputs(eps, contract_topics("mobile_so101"), "open_drawer", "mobile_so101")
    one_problem(probs, eps[0], "/gripper_controller/joint_trajectory")


def test_problems_per_episode(producer):
    # DATASET-v2 8.4: one problem string each, naming the episode: only the bad episodes are named
    p = producer()
    good = p.episode()
    bad1 = p.episode(result=False)
    bad2 = p.episode(drop=("/task/prompt",))
    eps, probs = check(p, good, bad1, bad2)
    assert probs and all(x.startswith("inputs:") for x in probs)
    assert not any(eps[0].where in x for x in probs), probs
    assert any(eps[1].where in x for x in probs) and any(eps[2].where in x and "/task/prompt" in x for x in probs)
    assert eps[0].duration_s == 20.004


def test_metadata_written_by_helper_is_rosbag2(tmp_path):
    # the fixture itself: metadata.yaml round-trips through bag_message_counts (guards the other tests)
    write_bag_metadata(tmp_path / "b", {"/a": 3, "/b": 0})
    assert M.bag_message_counts(Path(tmp_path / "b")) == {"/a": 3, "/b": 0}
