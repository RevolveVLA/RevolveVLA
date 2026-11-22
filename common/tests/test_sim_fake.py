"""sim_fake: the same loop as the real session, schema of the written files, fault injection."""
import json

import pytest

from revolve_common.sim_fake import FakeSession
from revolve_common.sim_session import READY_TIMEOUT, RelaunchPolicy, SimFault, run_with_relaunch

RESULT_KEYS = {"task", "scene", "robot", "episode", "sim_time_start", "passed", "failed_phase", "reason", "t_fail", "check",
               "condition", "value", "duration_s", "phases", "sanity_last", "rate_hz", "tick_ms", "contacts", "gz_log",
               "result_file", "bag"}


def sess(tmp_path, **kw):
    return FakeSession("kitchen_partnet", "mobile_panda", "open_fridge_door", tmp_path / "scene_0", "f-0", 110, **kw)


def episode(client, k=None):
    n = client.start_episode()
    res = client.wait_result(n)
    return n, res, client.wait_files(n)


def test_two_episodes_with_a_reset_write_the_contract_files(tmp_path):
    s = sess(tmp_path, passes=lambda n: n == 1)
    with s:
        s.start()
        c = s.connect()
        c.wait_ready()
        n0, r0, f0 = episode(c)
        c.reset()
        n1, r1, f1 = episode(c)
    assert (n0, n1) == (0, 1) and r0 == f0 and r1 == f1 and set(r0) == RESULT_KEYS
    assert not r0["passed"] and r0["failed_phase"] == "approach" and r1["passed"] and r1["failed_phase"] is None
    assert sorted(p.name for p in (tmp_path / "scene_0").iterdir()) == \
        ["episode_0.json", "episode_0_bag", "episode_1.json", "episode_1_bag"]
    assert json.loads((tmp_path / "scene_0/episode_1.json").read_text())["result_file"] == str(tmp_path / "scene_0/episode_1.json")
    assert r0["scene"] == "kitchen_partnet" and c.resets() == 1


def test_a_relaunch_continues_the_numbering_and_start_is_refused_while_running(tmp_path):
    s = sess(tmp_path)
    s.start()
    c = s.connect()
    episode(c)
    c.start_episode()
    with pytest.raises(Exception, match="running"):
        c.start_episode()
    s.start()
    assert episode(s.connect())[0] == 1 and s.launches == 2 and s.closed == 1


def test_injected_faults_drive_run_with_relaunch(tmp_path):
    s = sess(tmp_path, faults={"wait_ready": [READY_TIMEOUT]})
    policy = RelaunchPolicy()

    def work(session):
        c = session.connect()
        c.wait_ready()
        return episode(c)[0]
    assert run_with_relaunch(s, work, policy, log=lambda m: None) == 0
    assert policy.faults == [READY_TIMEOUT] and s.launches == 2 and not s.started


def test_check_fault_injection(tmp_path):
    s = sess(tmp_path, faults={"check": ["sim_exit"]})
    assert s.check() is None          # not started
    s.start()
    assert s.check().kind == "sim_exit" and s.check() is None
