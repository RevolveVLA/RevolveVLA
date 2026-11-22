"""sim_session: the revolve sim argv (DATAGEN 4.1), DomainPool, faults (sim exit, clock fault), close, the client
container argv, relaunch rules. Fake docker + fake popen: no docker, no sim."""
import subprocess
import sys

import pytest

from revolve_common import sim_session as ss
from revolve_common.sim_session import (ClientContainer, DomainPool, RelaunchPolicy, SimFault, SimInfraError,
                                        SimSession, run_with_relaunch, stop_sims, wait_files)


class FakeDocker:
    def __init__(self, names=(), domains=()):
        self._names, self._domains, self.removed = list(names), set(domains), []

    def names(self):
        return self._names

    def domains(self):
        return self._domains

    def rm(self, name):
        self.removed.append(name)


class FakeProc:
    def __init__(self, rc=None):
        self.rc, self.killed = rc, False

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        return self.rc

    def kill(self):
        self.killed = True


def session(tmp_path, **kw):
    popped = []

    def popen(argv, **k):
        popped.append((argv, k))
        return FakeProc()
    s = SimSession("runs/x/scene.yaml", "mobile_panda", "open_fridge_door", tmp_path / "scene_7", "dg-0", 120,
                   log=tmp_path / "sim.log", docker=kw.pop("docker", FakeDocker()), popen=popen, **kw)
    return s, popped


def test_argv_is_exactly_the_contract(tmp_path):
    s, _ = session(tmp_path)
    assert s.argv() == [str(ss.REPO / "revolve"), "sim", "--scene", "runs/x/scene.yaml", "--robot", "mobile_panda",
                        "--task", "open_fridge_door", "--no-autostart", "--record", "images", "--out",
                        str((tmp_path / "scene_7").resolve()), "--name", "dg-0", "--domain-id", "120"]
    g, _ = session(tmp_path, record="no-images", gui=True)
    assert g.argv()[g.argv().index("--record") + 1] == "no-images" and g.argv()[-1] == "--gui"
    with pytest.raises(ValueError):
        session(tmp_path, record="video")


def test_start_launches_in_the_repo_in_its_own_session_and_logs(tmp_path):
    s, popped = session(tmp_path)
    s.start()
    argv, kw = popped[0]
    assert argv == s.argv() and kw["cwd"] == ss.REPO and kw["start_new_session"] and kw["stderr"] == subprocess.STDOUT
    assert "--- " in (tmp_path / "sim.log").read_text()


def test_close_is_idempotent_and_removes_the_container(tmp_path):
    d = FakeDocker()
    s, _ = session(tmp_path, docker=d)
    s.close()                                   # before start: nothing
    s.start()
    s.close()
    s.close()
    assert d.removed == ["dg-0"] and s.check() is None
    with s:
        s.start()
    assert d.removed == ["dg-0", "dg-0"]


def test_start_again_closes_the_previous_launch(tmp_path):
    d = FakeDocker()
    s, popped = session(tmp_path, docker=d)
    s.start()
    s.start()
    assert d.removed == ["dg-0"] and len(popped) == 2


def test_check_sim_exit(tmp_path):
    s, _ = session(tmp_path)
    s.start()
    assert s.check() is None
    s._proc.rc = 3
    f = s.check()
    assert f.kind == ss.SIM_EXIT and "3" in f.detail


def test_check_clock_fault_needs_three_lines_counted_across_polls(tmp_path):
    s, _ = session(tmp_path)
    s.start()
    log = tmp_path / "sim.log"
    with open(log, "a") as f:
        f.write("[ros2_control] No clock received, using time argument instead!\n" * 2)
    assert s.check() is None                                    # a race at bring-up
    with open(log, "a") as f:
        f.write("noise\nNo clock received, using time argument instead!\n")
    assert s.check().kind == ss.CLOCK_FAULT


def test_a_relaunch_forgets_the_clock_lines_of_the_launch_before(tmp_path):
    s, _ = session(tmp_path)
    s.start()
    with open(tmp_path / "sim.log", "a") as f:
        f.write("No clock received\n" * 5)
    assert s.check() is not None
    s.start()
    assert s.check() is None


def test_domain_pool_distinct_names_and_domains():
    p = DomainPool("run1", range(10, 14))
    got = [p.lease() for _ in range(3)]
    assert [n for n, _ in got] == ["run1-0", "run1-1", "run1-2"] and len({d for _, d in got}) == 3
    p.release("run1-1")
    assert p.lease()[0] == "run1-1"


def test_domain_pool_skips_running_containers_and_runs_out():
    p = DomainPool("r", range(10, 13), FakeDocker(names=["r-0", "r-1-client"], domains={10}))
    assert p.lease() == ("r-2", 11)
    assert p.lease()[1] == 12
    with pytest.raises(SimInfraError):
        p.lease()


def test_stop_sims_removes_the_prefix_only():
    d = FakeDocker(names=["run1-0", "run1-0-client", "run10-0", "other"])
    assert stop_sims("run1", d) == 2 and d.removed == ["run1-0", "run1-0-client"]


def test_client_argv_mirrors_the_contract_test(tmp_path):
    c = ClientContainer("dg-0-client", 120, ["python3", "/repo/x.py", "--a", "1"], tmp_path / "o", pythonpath=["/repo/y"],
                        mounts=[tmp_path / "m"], docker=FakeDocker())
    a = c.argv()
    r, o, m = ss.REPO, (tmp_path / "o").resolve(), tmp_path / "m"
    assert a[:7] == ["docker", "run", "--rm", "--name", "dg-0-client", "--network", "host"]
    for pair in (["-e", "ROS_DOMAIN_ID=120"], ["-e", "FASTDDS_BUILTIN_TRANSPORTS=UDPv4"],
                 ["-e", "PYTHONPATH=/repo/common:/repo/y"], ["-v", f"{r}:/repo:ro"], ["-v", f"{o}:{o}"],
                 ["-v", f"{m}:{m}"], ["-v", f"{r}/ThirdParty/gz_kura/install:/opt/gz_ws/install:ro"],
                 ["-v", f"{r}/ThirdParty/ros_gz_ws/install:/opt/ros_gz_ws/install:ro"]):
        assert any(a[i:i + 2] == pair for i in range(len(a))), pair
    assert "--gpus" not in a
    assert a[a.index(ss.SIM_IMAGE):] == [ss.SIM_IMAGE, "bash", "-c", 'exec "$@"', "client", "python3", "/repo/x.py",
                                         "--a", "1"]


def run_client(tmp_path, code, watch=None, **kw):
    d = FakeDocker()
    c = ClientContainer("c", 1, [], tmp_path, docker=d)
    c.argv = lambda: [sys.executable, "-c", code]
    return c, d, c.run(tmp_path / "c.log", watch=watch, poll_s=0.05, **kw)


def test_client_run_returns_rc_and_logs(tmp_path):
    _, _, rc = run_client(tmp_path, "print('hi'); raise SystemExit(4)")
    assert rc == 4 and "hi" in (tmp_path / "c.log").read_text()


def test_client_exit_code_is_a_fault(tmp_path):
    with pytest.raises(SimFault) as e:
        run_client(tmp_path, f"raise SystemExit({ss.EXIT_CODES[ss.RESET_FAILED]})")
    assert e.value.kind == ss.RESET_FAILED


def test_watch_fault_removes_the_client(tmp_path):
    d = FakeDocker()
    c = ClientContainer("c", 1, [], tmp_path, docker=d)
    c.argv = lambda: [sys.executable, "-c", "import time; time.sleep(30)"]
    c.STOP_WAIT_S = 0.2                 # the fake docker kills nothing
    with pytest.raises(SimFault) as e:
        c.run(tmp_path / "c.log", watch=lambda: SimFault(ss.CLOCK_FAULT), poll_s=0.05)
    assert e.value.kind == ss.CLOCK_FAULT and d.removed == ["c"]


def test_client_timeout(tmp_path):
    d = FakeDocker()
    c = ClientContainer("c", 1, [], tmp_path, docker=d)
    c.argv = lambda: [sys.executable, "-c", "import time; time.sleep(30)"]
    c.STOP_WAIT_S = 0.2                 # the fake docker kills nothing
    with pytest.raises(subprocess.TimeoutExpired):
        c.run(tmp_path / "c.log", timeout=0.3)
    assert d.removed == ["c"]


def test_wait_files(tmp_path):
    (tmp_path / "episode_2.json").write_text('{"passed": true}')
    assert wait_files(tmp_path, 2) == {"passed": True}
    with pytest.raises(SimFault) as e:
        wait_files(tmp_path, 3, timeout_s=0.1)
    assert e.value.kind == ss.FILES_TIMEOUT


class Sess:
    def __init__(self):
        self.name, self.log = "s", []

    def start(self):
        self.log.append("start")

    def close(self):
        self.log.append("close")


def test_relaunch_after_a_fault_then_success_counted():
    s, policy, calls = Sess(), RelaunchPolicy(2), []

    def work(_):
        calls.append(1)
        if len(calls) == 1:
            raise SimFault(ss.READY_TIMEOUT)
        return "ok"
    assert run_with_relaunch(s, work, policy, log=lambda m: None) == "ok"
    assert s.log == ["start", "start", "close"] and policy.faults == [ss.READY_TIMEOUT]


def test_relaunch_gives_up_after_max_and_still_closes():
    s, policy = Sess(), RelaunchPolicy(2)

    def work(_):
        raise SimFault(ss.RESET_FAILED, "x")
    with pytest.raises(SimInfraError):
        run_with_relaunch(s, work, policy, log=lambda m: None)
    assert s.log.count("start") == 3 and s.log[-1] == "close" and len(policy.faults) == 2


def test_other_errors_pass_through_and_close():
    s = Sess()
    with pytest.raises(KeyError):
        run_with_relaunch(s, lambda _: {}["x"])
    assert s.log == ["start", "close"]


def test_fault_exit_codes_round_trip():
    for kind in ss.EXIT_CODES:
        assert SimFault.from_exit(SimFault(kind).exit_code).kind == kind
    assert SimFault.from_exit(0) is None and SimFault.from_exit(1) is None
