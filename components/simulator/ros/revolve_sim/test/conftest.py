"""Fixtures for the simulator tests on a launched sim (host side, `./revolve test simulator`): each test module
starts one fresh `revolve sim` container, runs the check_*.py scripts inside it with `docker exec` and asserts on their JSON.

  T1 bring-up (test_bringup.py): the world from scene.yaml, the robot at robot_start, every controller active, the
     cameras at rate with non-empty images, the finger contact topics, no gz/ROS error lines.
  T2 control (test_control.py): base drive / strafe / rotate tracking against the gz ground truth, arm joint targets
     reached and held (effort JTC), gripper open / close, finger contacts.
  T3 evaluator negatives (test_evaluator_idle.py, test_evaluator_push.py; the sim with the task, episodes started
     on /evaluator/start): an idle episode fails `approach` on its timeout; driving the base into the fridge fails
     on a sanity check; the evaluator's metric values at the start match the scene geometry.
  Unit tests of the evaluator logic without a sim: test_episode_unit.py.
A test module sets SIM_ARGS (extra `revolve sim` args, e.g. --task) for its sim.

Env: REVOLVE_TEST_SCENE / REVOLVE_TEST_ROBOT (default kitchen_partnet / mobile_panda), REVOLVE_TEST_LOGS (default
runs/test_simulator/<time>). Needs `./revolve build simulator` and the scene's assets.
"""
import json
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[5]
SCENE = os.environ.get("REVOLVE_TEST_SCENE", "kitchen_partnet")
ROBOT = os.environ.get("REVOLVE_TEST_ROBOT", "mobile_panda")
TEST_IN = "/repo/components/simulator/ros/revolve_sim/test"
# gz / ROS problem lines; the known benign ones: MuJoCo's NaN-in-CTRL line at load (a WARNING, not matched)
PROBLEM = re.compile(r"\[(error|Err|ERROR|FATAL|critical)\]|Traceback|not added")

def sh(name, cmd, timeout):
    """Run cmd in the container (bash: the image's ROS env)."""
    return subprocess.run(["docker", "exec", name, "bash", "-c", cmd], capture_output=True, text=True,
                          timeout=timeout)


def result(name, script, args, timeout, log):
    """Run a check script in the container -> its RESULT JSON."""
    r = sh(name, f"python3 {TEST_IN}/{script} --scene {SCENE} --robot {ROBOT} {args}", timeout)
    log.write_text(r.stdout + r.stderr)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"{script} printed no RESULT (rc {r.returncode}): see {log}"
    return json.loads(lines[-1][len("RESULT "):])


@pytest.fixture(scope="module")
def sim(request):
    if shutil.which("docker") is None:
        pytest.skip("needs docker")
    logs = Path(os.environ.get("REVOLVE_TEST_LOGS", REPO / "runs" / "test_simulator" / time.strftime("%Y%m%d_%H%M%S")))
    logs = logs / request.module.__name__.split(".")[-1]
    logs.mkdir(parents=True, exist_ok=True)
    name = f"revolve-sim-test-{os.getpid()}"
    rdir = REPO / "specs" / "robots" / ROBOT
    cfile = rdir / yaml.safe_load((rdir / "robot.yaml").read_text())["controllers"]
    controllers = [k for k, v in yaml.safe_load(cfile.read_text())["controller_manager"]["ros__parameters"].items()
                   if isinstance(v, dict) and "type" in v]
    env = {**os.environ, "ROS_DOMAIN_ID": str(random.randint(10, 99))}
    launch_log = logs / "launch.log"
    with open(launch_log, "w") as f:
        p = subprocess.Popen([str(REPO / "revolve"), "sim", "--scene", SCENE, "--robot", ROBOT, "--name", name,
                              *getattr(request.module, "SIM_ARGS", [])],
                             stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO, start_new_session=True)
    try:
        deadline = time.time() + 240
        while True:
            assert p.poll() is None, f"revolve sim exited early: see {launch_log}"
            text = launch_log.read_text(errors="replace")
            assert "Failed loading controller" not in text and "Could not configure" not in text, \
                f"a controller failed to load: see {launch_log}"
            if "Configured and activated" in text:
                r = sh(name, "timeout 20 ros2 control list_controllers", 30)
                states = re.findall(r"\b(active|inactive|unconfigured)\b", r.stdout)
                if len(states) == len(controllers) and all(s == "active" for s in states):
                    break
            assert time.time() < deadline, f"controllers not active after 240 s: see {launch_log}"
            time.sleep(2)
        yield {"name": name, "logs": logs, "launch_log": launch_log}
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
