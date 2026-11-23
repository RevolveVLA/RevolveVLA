"""sim_client logic without ROS: states and results fed by hand, the Trigger services stubbed."""
import json
import threading
import time

import pytest

from revolve_common import sim_session as ss
from revolve_common.sim_client import EpisodeRefused, SimClient


def state(resets=0, ready=True, t=1.0):
    return json.dumps({"t": t, "ready": ready, "resets": resets, "episode": None, "phase": None, "values": {}})


class Client(SimClient):
    def __init__(self, replies, **kw):
        super().__init__(**kw)
        self.replies, self.calls = replies, []

    def _trigger(self, service, wall_s):
        self.calls.append(service)
        return self.replies[service]


def test_wait_ready():
    c = Client({}, out=None)
    with pytest.raises(ss.SimFault) as e:
        c.wait_ready(0.1)
    assert e.value.kind == ss.READY_TIMEOUT
    c.on_state(state(ready=False))
    with pytest.raises(ss.SimFault):
        c.wait_ready(0.1)
    c.on_state(state())
    c.wait_ready(0.1)


def test_start_episode_parses_the_number_and_refusal():
    c = Client({"/evaluator/start": (True, "episode 4")})
    assert c.start_episode() == 4
    c.replies["/evaluator/start"] = (False, "episode 4 is running")
    with pytest.raises(EpisodeRefused, match="running"):
        c.start_episode()


def test_reset_waits_for_more_resets_and_ready():
    c = Client({"/simulator/reset": (True, "reset in 1.7 s")})
    c.on_state(state(resets=2))

    def feed():
        time.sleep(0.1)
        c.on_state(state(resets=3, ready=False))        # reset seen, not ready yet
        time.sleep(0.1)
        c.on_state(state(resets=3))
    threading.Thread(target=feed).start()
    c.reset(2)
    assert c.calls == ["/simulator/reset"] and c.resets() == 3


def test_reset_false_reply_and_timeout_are_reset_failed():
    c = Client({"/simulator/reset": (False, "step timed out")})
    c.on_state(state(resets=0))
    with pytest.raises(ss.SimFault, match="step timed out") as e:
        c.reset(0.1)
    assert e.value.kind == ss.RESET_FAILED
    c.replies["/simulator/reset"] = (True, "ok")
    with pytest.raises(ss.SimFault) as e:               # no state with resets > 0 ever arrives
        c.reset(0.1)
    assert e.value.kind == ss.RESET_FAILED


def test_an_old_ready_state_does_not_end_the_reset_wait():
    c = Client({"/simulator/reset": (True, "ok")})
    for r in (1, 1, 1):
        c.on_state(state(resets=r))
    with pytest.raises(ss.SimFault):
        c.reset(0.1)


def test_wait_result_by_episode_and_a_late_result_of_the_one_before():
    c = Client({})
    c.on_result(json.dumps({"episode": 0, "passed": True}))     # transient local: delivered at subscription
    with pytest.raises(ss.SimFault) as e:
        c.wait_result(1, 0.1)
    assert e.value.kind == ss.RESULT_TIMEOUT
    c.on_result(json.dumps({"episode": 1, "passed": False}))
    assert c.wait_result(1, 0.1)["passed"] is False


def test_wait_files(tmp_path):
    c = Client({}, out=tmp_path)
    (tmp_path / "episode_1.json").write_text('{"episode": 1}')
    assert c.wait_files(1) == {"episode": 1}
