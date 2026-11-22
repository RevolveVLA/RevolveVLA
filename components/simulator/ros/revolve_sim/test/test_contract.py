"""The datagen <-> simulator contract (components/datagen/DATAGEN-v2.md), as datagen uses it: a scene instance
DIR/scene.yaml (the scene with the robot start moved, as a randomizer writes it), the sim from the CLI (`revolve sim
--scene DIR/scene.yaml --task T --no-autostart --record images --out DIR --name N --domain-id D`, the conftest
fixture; the robot must start at the instance's robot_start), the
client (check_contract.py) in its own container from the sim image on the host network with ROS_DOMAIN_ID=D, using only
the documented interface: two episodes with a reset between them, then the files in DIR: two result JSONs, two gz log
copies and two closed bags, every frozen topic (config/record.yaml) with messages, the cameras JPEG at ~30 Hz,
/tf_static and /robot_description, controller_state at >= 30 Hz, and a 30 Hz action series (controller_state reference +
cmd_vel) from each bag. DIR (runs/contract_<pid>/, ~100 MB of images per episode) is removed when the test passes; its
sizes and rates go into the log folder's contract.json."""
import json
import os
import shutil
import subprocess

import robot_cases
import pytest
import yaml
from conftest import REPO, SCENE, TEST_IN, problem_lines

ROBOTS = ["mobile_panda"]
TASK = robot_cases.TASKS["mobile_panda"]["task"]
OUT = REPO / "runs" / f"contract_{os.getpid()}"
DOMAIN = 102 + os.getpid() % 100            # outside the range revolve sim derives from container names (1..101)
INSTANCE = OUT / "scene.yaml"
SHIFT_X = 0.10                              # the instance's robot start: the scene's, 0.10 m to +x (open floor)
SIM_ARGS = ["--scene", str(INSTANCE), "--task", TASK, "--no-autostart", "--record", "images", "--out", str(OUT),
            "--domain-id", str(DOMAIN)]
IMAGE = os.environ.get("REVOLVE_SIM_IMAGE", "revolve-simulator:latest")


@pytest.fixture(scope="module", autouse=True)
def scene_instance():
    """DIR/scene.yaml before the sim starts (autouse: before the sim fixture)."""
    d = yaml.safe_load((REPO / "specs" / "scenes" / SCENE / "scene.yaml").read_text())
    d["robot_start"]["base"][0] += SHIFT_X
    OUT.mkdir(parents=True, exist_ok=True)
    INSTANCE.write_text(yaml.safe_dump(d, sort_keys=False))
    return d


def client(sim, timeout=600):
    """check_contract.py in a client container (as datagen's): host network, the sim's domain, --out mounted."""
    name = f"{sim['name']}-client"
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "host", "-u", f"{os.getuid()}:{os.getgid()}",
           "-e", "HOME=/tmp", "-e", f"ROS_DOMAIN_ID={DOMAIN}", "-e", "FASTDDS_BUILTIN_TRANSPORTS=UDPv4",
           "-e", "PYTHONPATH=/repo/common", "-v", f"{REPO}:/repo:ro", "-v", f"{OUT}:{OUT}",
           "-v", f"{REPO}/ThirdParty/gz_kura/install:/opt/gz_ws/install:ro",
           "-v", f"{REPO}/ThirdParty/ros_gz_ws/install:/opt/ros_gz_ws/install:ro", IMAGE, "bash", "-c",
           f"python3 {TEST_IN}/check_contract.py --scene {INSTANCE} --robot {sim['robot']} --task {TASK} --out {OUT}"]
    log = sim["logs"] / "check_contract.log"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    log.write_text(r.stdout + r.stderr)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"check_contract.py printed no RESULT (rc {r.returncode}): see {log}"
    return json.loads(lines[-1][len("RESULT "):])


def test_contract(sim):
    r = client(sim)
    (sim["logs"] / "contract.json").write_text(json.dumps(r, indent=1))
    assert all(ok for ok, _ in r["start"]), r["start"]
    assert r["arm_goal"] == 0 and r["hand_goal"] == 0, (r["arm_goal"], r["hand_goal"])
    assert r["reset"]["success"], r["reset"]
    eps = r["episodes"]
    assert len(eps) == 2 and eps[1]["result"]["episode"] == eps[0]["result"]["episode"] + 1
    assert sorted(r["files"]) == sorted(["scene.yaml", *(f"episode_{e['result']['episode']}{s}" for e in eps
                                                        for s in (".json", "_bag", "_gz.log"))])
    assert r["start_err_m"] <= 0.01 and r["reset"]["start_err_m"] <= 0.01, r     # at the instance's robot_start
    for ep in eps:
        res = ep["result"]
        assert res["reason"] == "timeout_s" and res["rate_hz"] >= 95.0, res
        assert ep["in_out_dir"] and ep["bag_closed_at_result_file"], ep
        assert not ep["missing"] and not ep["extra"] and not ep["empty"], (ep["missing"], ep["extra"], ep["empty"])
        assert ep["counts"]["/tf_static"] >= 1 and ep["counts"]["/robot_description"] >= 1, ep["counts"]
        assert ep["result_in_bag"] == [res["episode"]], ep["result_in_bag"]
        assert ep["jpeg"]
        for t, hz in ep["rates_hz"].items():
            if t.endswith("/image/compressed"):
                assert 27.0 <= hz <= 33.0, (t, hz)
            elif t.endswith("/controller_state"):
                assert hz >= 30.0, (t, hz)
        s = ep["series"]
        assert s["finite"] and abs(s["hz"] - 30.0) < 0.1 and s["n"] >= 30 * (res["duration_s"] - 1.0), s
    s = eps[0]["series"]
    assert abs(s["cmd_vx_min"] + 0.2) < 0.005, s                 # the ramped segment's peak
    assert s["arm_ref_err_at_goal_end"] < 1e-3, s                # FollowJointTrajectory goal: the reference at the end
    assert abs(s["hand_ref_at_goal_end"] - robot_cases.CONTROL["mobile_panda"]["closed"]) < 1e-3, s
    assert s["stream_joint1_range"] > 0.1 and s["stream_last_err"] < 1e-3, s   # the streamed points
    assert not problem_lines(sim["launch_log"]), problem_lines(sim["launch_log"])[:5]
    shutil.rmtree(OUT)
