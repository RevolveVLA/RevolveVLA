"""dataset.port.LogProblems: Rosetta's porter only logs what it does to a bad stream and carries on, so its log is read
for problems (DATASET-v2.md section 8.4). The log lines here are Rosetta's exact strings at the pinned commit
(rosetta_ros/ingest.py, offline/bag_frames.py, offline/port.py). Host only: port.py imports nothing heavy at module
level."""
import importlib
import logging
import sys

import pytest

HAND = "/panda_hand_controller/joint_trajectory"
ARM = "/panda_arm_controller/joint_trajectory"
CMD_VEL = "/omni_wheel_drive_controller/cmd_vel"


@pytest.fixture
def root_logger():
    """The root logger at INFO (as while Rosetta's porter runs); handlers added by the test removed afterwards."""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    root.setLevel(logging.INFO)
    yield root
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
    root.setLevel(level)


def attach(root, names):
    from dataset.port import LogProblems
    h = LogProblems(list(names))
    root.addHandler(h)
    return h


def episode_line(i, n):
    logging.info("%d / %d episodes processed", i, n)                   # offline/port.py write_dataset


def decode_failed(key="action", topic=HAND):
    # rosetta_ros/ingest.py StreamIngest.ingest (WARNING via logging.warning offline)
    e = ValueError("JointTrajectoryPoint name 'panda_finger_joint1' not in message")
    text = f"Decode failed for '{key}' on {topic} ({type(e).__name__}: {e}); dropping message"
    logging.warning(text)
    return text


def timeline_missing(key="action", topic=HAND, timeline="receive"):
    text = f"Message on '{key}' ({topic}) is missing its '{timeline}' timeline; dropping message"
    logging.warning(text)
    return text


def zero_filled(key="action", topic=HAND):
    logging.warning("Topic %s not in bag, %s will be zero-filled", topic, key)     # offline/bag_frames.py
    return f"Topic {topic} not in bag, {key} will be zero-filled"


def failed(name="episode_1_bag", err="Bag has no messages"):
    logging.error("  -> FAILED %s: %s", name, err)                                 # offline/port.py
    return f"  -> FAILED {name}: {err}"


PROBLEM_LINES = {"decode": decode_failed, "timeline": timeline_missing, "zero": zero_filled, "failed": failed}


def test_port_imports_nothing_heavy():
    mod = importlib.import_module("dataset.port")
    assert isinstance(mod.LogProblems(["a"]), logging.Handler)
    heavy = [m for m in ("lerobot", "torch", "rclpy", "rosbag2_py", "rosetta", "rosetta_ros", "cv2")
             if m in sys.modules]
    assert not heavy, heavy


@pytest.mark.parametrize("kind", list(PROBLEM_LINES))
def test_each_rosetta_problem_line(root_logger, kind):
    # each of the four problem lines becomes one `port:` problem naming its episode and quoting the line
    names = ["run/manifest.json entry 0", "run/manifest.json entry 3"]
    h = attach(root_logger, names)
    episode_line(0, 2)
    episode_line(1, 2)
    text = PROBLEM_LINES[kind]()
    hits = [p for p in h.out if text.strip() in p]
    assert hits, (text, h.out)
    for p in hits:
        assert p.startswith("port:") and names[1] in p and names[0] not in p, p


def test_decode_failure_and_timeline_drop_in_one_episode(root_logger):
    # a stream's decode failure is logged once per stream and must not be lost; a timeline drop on another stream of
    # the same episode is a second problem
    names = ["m.json entry 0"]
    h = attach(root_logger, names)
    episode_line(0, 1)
    t1 = decode_failed()
    t2 = timeline_missing(key="action", topic=ARM)
    assert any(t1 in p and p.startswith("port:") and names[0] in p for p in h.out), h.out
    assert any(t2 in p and p.startswith("port:") and names[0] in p for p in h.out), h.out


def test_attribution_follows_the_episode_lines(root_logger):
    # every problem after "<i> / <n> episodes processed" belongs to episode i: problems in episodes 0 and 2, none in 1
    names = ["alpha.json entry 0", "bravo.json entry 1", "charlie.json entry 2"]
    h = attach(root_logger, names)
    episode_line(0, 3)
    a = decode_failed(topic=CMD_VEL)
    logging.info("  -> %d frames from %s", 601, "episode_0_bag")
    episode_line(1, 3)
    logging.info("  -> %d frames from %s", 602, "episode_1_bag")
    episode_line(2, 3)
    c = failed("episode_2_bag", "boom")
    pa = [p for p in h.out if a in p]
    pc = [p for p in h.out if c.strip() in p]
    assert pa and all(names[0] in p and names[1] not in p and names[2] not in p for p in pa), h.out
    assert pc and all(names[2] in p and names[0] not in p and names[1] not in p for p in pc), h.out
    assert not any(names[1] in p for p in h.out), h.out


def test_several_problems_in_one_episode_all_listed(root_logger):
    names = ["m.json entry 0"]
    h = attach(root_logger, names)
    episode_line(0, 1)
    texts = [decode_failed(topic=HAND), decode_failed(topic=ARM), zero_filled(topic=CMD_VEL)]
    for t in texts:
        assert any(t in p and p.startswith("port:") for p in h.out), (t, h.out)


def test_benign_lines_are_not_problems(root_logger):
    # recoveries, warmup (both levels), the different-contract note, Completed, frame counts, plain info lines and the
    # episode lines themselves
    names = ["m.json entry 0", "m.json entry 1"]
    h = attach(root_logger, names)
    logging.info("Found %d bags in %s", 2, "/tmp/staging")
    logging.info("Shard %d/%d: processing %d bags", 0, 1, 2)
    logging.warning("Bag was recorded with a different contract than --contract; using --contract for decoding and "
                    "embedding.")
    episode_line(0, 2)
    logging.info("%s: skipped %d warmup ticks (%.2fs)", "episode_0_bag", 2, 0.07)
    logging.warning("%s: skipped %d warmup ticks (%.2fs) waiting for observation streams; stream delays (latest "
                    "sample at warmup): %s", "episode_0_bag", 120, 4.0, "observation.images.wrist: +4.00s")
    logging.info(f"Timeline 'receive' recovered for 'action' ({HAND})")
    logging.info(f"Decode recovered for 'action' ({HAND})")
    logging.info("  -> %d frames from %s", 601, "episode_0_bag")
    episode_line(1, 2)
    logging.info("  -> %d frames from %s", 600, "episode_1_bag")
    logging.info("Completed: %d/%d episodes (%d failed) in %.1fs", 2, 2, 0, 12.3)
    assert h.out == [], h.out


def test_failed_line_with_the_run_summary(root_logger):
    # the FAILED line is a problem of its episode; the end-of-run summary ("Completed", "  - name: error") may or may
    # not add one, but nothing is attributed to the episode that succeeded
    names = ["m.json entry 0", "m.json entry 1"]
    h = attach(root_logger, names)
    episode_line(0, 2)
    logging.info("  -> %d frames from %s", 601, "episode_0_bag")
    episode_line(1, 2)
    text = failed("episode_1_bag", "Warmup never completed")
    logging.info("Completed: %d/%d episodes (%d failed) in %.1fs", 1, 2, 1, 9.0)
    logging.warning("  - %s: %s", "episode_1_bag", "Warmup never completed")
    hits = [p for p in h.out if text.strip() in p]
    assert hits and all(p.startswith("port:") and names[1] in p for p in hits), h.out
    assert not any(names[0] in p for p in h.out), h.out
