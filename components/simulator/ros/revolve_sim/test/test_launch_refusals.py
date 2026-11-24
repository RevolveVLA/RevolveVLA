"""A scene instance through the launched simulator (docker, the simulator image), not the conftest `sim` fixture:
a valid instance starts, and the launch itself still refuses a robot_start.arm that does not fit the robot when
`./revolve sim`'s host check is bypassed (its docker command run directly, as a client could). The host refusals:
test_cli.py."""
import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
import time

import pytest
from host_cases import REPO, pose, write_instance

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="needs docker")

TIMEOUT_S = 240


def revolve_cli():
    """The ./revolve script as a module: its docker command helpers are the only way to start the launch without
    `revolve sim`'s host check, which this backstop test must bypass."""
    loader = importlib.machinery.SourceFileLoader("revolve_cli", str(REPO / "revolve"))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


def test_launch_refuses_an_arm_that_does_not_fit(tmp_path):
    cli = revolve_cli()
    path = write_instance(tmp_path / "scene.yaml", arm=[*pose("mobile_so101", "ready"), 0.0])     # 6 values, 5 joints
    name = f"revolve-sim-refusal-{os.getpid()}"
    launch = ["ros2", "launch", "revolve_sim", "sim.launch.py", f"scene:={path}", "robot:=mobile_so101", "gui:=false"]
    cmd = [*cli.sim_docker_args(name, False, cli.sim_domain(name), None, [path]), cli.SIM_IMAGE,
           "bash", "-c", 'exec "$@"', "sim", *launch]
    try:
        r = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           timeout=TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"the launch still runs after {TIMEOUT_S} s: the instance was not refused")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    assert r.returncode == 1, f"exit code {r.returncode}, output:\n{r.stdout[-3000:]}"
    assert "robot_start.arm has 6 joints, robot mobile_so101 5" in r.stdout, r.stdout[-3000:]


def test_valid_instance_runs(tmp_path):
    name = f"revolve-sim-refusal-{os.getpid()}-valid"
    log = tmp_path / "launch.log"
    cmd = [str(REPO / "revolve"), "sim", "--scene", str(write_instance(tmp_path / "scene.yaml", arm="ready")),
           "--robot", "mobile_so101", "--name", name]
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + TIMEOUT_S
        while True:
            text = log.read_text(errors="replace")
            assert p.poll() is None, f"revolve sim exited (code {p.returncode}) on a valid instance:\n{text[-3000:]}"
            if "Configured and activated" in text:      # the robot is up (controllers activated)
                break
            assert time.time() < deadline, f"controllers not up after {TIMEOUT_S} s: see {log}"
            time.sleep(2)
        assert p.poll() is None, "revolve sim exited right after its controllers came up"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
