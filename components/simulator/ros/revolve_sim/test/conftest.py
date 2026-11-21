"""Fixtures for the simulator tests on a launched sim (host side, `./revolve test simulator`): each test module
starts one fresh `revolve sim` container, runs the check_*.py scripts inside it with `docker exec` and asserts on their JSON.

  T1 bring-up (test_bringup.py): the world from scene.yaml, the robot at robot_start, every controller active, the
     cameras at rate with non-empty images, the finger contact topics, no gz/ROS error lines.
  T2 control (test_control.py): base drive / strafe / rotate tracking against the gz ground truth, arm joint targets
     reached and held (effort JTC), gripper open / close, finger contacts.
  T3 evaluator negatives (test_evaluator_idle.py, test_evaluator_push.py; the sim with the robot's task of
     robot_cases.TASKS, episodes started on /evaluator/start): an idle episode fails `approach` on its timeout;
     driving the base into the fridge / the counter fails on a sanity check; the evaluator's metric values at the
     start match the scene geometry.
  T4 positive (test_positive.py, marker `slow`): MoveIt 2 (moveit_py) plans the arm, the base gets open-loop
     ramped TwistStamped segments (test code only): open_fridge_door (mobile_panda, check_open_door.py) and
     open_drawer (mobile_so101_xl, check_open_drawer.py) pass approach, grasp and open.
  Reset (test_reset.py) and the datagen contract (test_contract.py: a client in its own container, two episodes,
     the recorded files).
  Unit tests of the evaluator logic without a sim: test_episode_unit.py.
A launch whose controller manager logs "No clock received" once a second (it never gets /clock; rare, cause not
pinned) fails at once with that message, before the test runs (no retry).
A test module sets ROBOTS (the robots it runs on, one fresh sim each; default mobile_panda; T1 and T2: every robot
of robot_cases.ROBOTS) and SIM_ARGS (extra `revolve sim` args, e.g. --task, or a dict of them per robot) for its sim;
with --task the fixture adds `--record no-images` unless SIM_ARGS has --record.

Env: REVOLVE_TEST_SCENE (default kitchen_partnet), REVOLVE_TEST_ROBOT (comma-separated: only these of each module's
ROBOTS), REVOLVE_TEST_LOGS (default runs/test_simulator/<time>). Needs `./revolve build simulator` and the scene's
assets.
"""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[5]
SCENE = os.environ.get("REVOLVE_TEST_SCENE", "kitchen_partnet")
ONLY_ROBOTS = [r for r in os.environ.get("REVOLVE_TEST_ROBOT", "").split(",") if r]
TEST_IN = "/repo/components/simulator/ros/revolve_sim/test"
# gz / ROS problem lines; the known benign ones: MuJoCo's NaN-in-CTRL line at load (a WARNING, not matched)
PROBLEM = re.compile(r"\[(error|Err|ERROR|FATAL|critical)\]|Traceback|not added")
# A broken launch: the controller manager (in the gz process) never gets /clock and warns once a second for the
# whole run (rare, cause not pinned). One or two lines at start are a race with the first /clock.
NO_CLOCK, NO_CLOCK_LINES = "No clock received", 3

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: the long positive episodes (T4); deselect with -m 'not slow'")


def pytest_generate_tests(metafunc):
    if "robot" in metafunc.fixturenames:
        robots = getattr(metafunc.module, "ROBOTS", ["mobile_panda"])
        robots = [r for r in robots if not ONLY_ROBOTS or r in ONLY_ROBOTS]
        metafunc.parametrize("robot", robots, scope="module")


def sh(name, cmd, timeout):
    """Run cmd in the container (bash: the image's ROS env)."""
    return subprocess.run(["docker", "exec", name, "bash", "-c", cmd], capture_output=True, text=True,
                          timeout=timeout)


def result(sim, script, args, timeout, log):
    """Run a check script in the sim's container -> its RESULT JSON."""
    r = sh(sim["name"], f"python3 {TEST_IN}/{script} --scene {SCENE} --robot {sim['robot']} {args}", timeout)
    log.write_text(r.stdout + r.stderr)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"{script} printed no RESULT (rc {r.returncode}): see {log}"
    return json.loads(lines[-1][len("RESULT "):])


def clock_fault(launch_log):
    n = Path(launch_log).read_text(errors="replace").count(NO_CLOCK)
    assert n < NO_CLOCK_LINES, (f"broken sim launch: the controller manager logged {NO_CLOCK!r} {n} times (it never "
                                f"got /clock); not retried, see {launch_log}")


@pytest.fixture(scope="module")
def sim(request, robot):
    if shutil.which("docker") is None:
        pytest.skip("needs docker")
    logs = Path(os.environ.get("REVOLVE_TEST_LOGS", REPO / "runs" / "test_simulator" / time.strftime("%Y%m%d_%H%M%S")))
    logs = logs / request.module.__name__.split(".")[-1] / robot
    logs.mkdir(parents=True, exist_ok=True)
    name = f"revolve-sim-test-{os.getpid()}"
    rdir = REPO / "specs" / "robots" / robot
    cfile = rdir / yaml.safe_load((rdir / "robot.yaml").read_text())["controllers"]
    controllers = [k for k, v in yaml.safe_load(cfile.read_text())["controller_manager"]["ros__parameters"].items()
                   if isinstance(v, dict) and "type" in v]
    launch_log = logs / "launch.log"
    sim_args = getattr(request.module, "SIM_ARGS", [])
    if isinstance(sim_args, dict):
        sim_args = sim_args[robot]
    if "--task" in sim_args and "--record" not in sim_args:
        sim_args = [*sim_args, "--record", "no-images"]     # the bags without the camera images (size)
    with open(launch_log, "w") as f:
        p = subprocess.Popen([str(REPO / "revolve"), "sim", "--scene", SCENE, "--robot", robot, "--name", name,
                              *sim_args],
                             stdout=f, stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
    try:
        deadline = time.time() + 240
        while True:
            assert p.poll() is None, f"revolve sim exited early: see {launch_log}"
            text = launch_log.read_text(errors="replace")
            assert "Failed loading controller" not in text and "Could not configure" not in text, \
                f"a controller failed to load: see {launch_log}"
            clock_fault(launch_log)
            if "Configured and activated" in text:
                r = sh(name, "timeout 20 ros2 control list_controllers", 30)
                states = re.findall(r"\b(active|inactive|unconfigured)\b", r.stdout)
                if len(states) == len(controllers) and all(s == "active" for s in states):
                    break
            assert time.time() < deadline, f"controllers not active after 240 s: see {launch_log}"
            time.sleep(2)
        time.sleep(NO_CLOCK_LINES)      # the fault would log on
        clock_fault(launch_log)
        yield {"name": name, "robot": robot, "logs": logs, "launch_log": launch_log}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)   # kills the whole launch
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


def host_path(container_path):
    """A /repo/... path in the sim container -> the host path."""
    return REPO / Path(container_path).relative_to("/repo")


def problem_lines(launch_log):
    return [ln for ln in Path(launch_log).read_text(errors="replace").splitlines() if PROBLEM.search(ln)]
