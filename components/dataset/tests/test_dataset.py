"""Real conversions in revolve-dataset (see conftest.py for the fixtures). All jobs start at once:
- joint:  convert [demo, recovery] as turn 0 (polled over the wrapper's HTTP)
- loop:   turn 0 = convert [demo]; turn 1 = launch.turn_dataset: convert [recovery] -> dataset_new, merge after
          turn 0 -> turn_1/dataset (the manager's per-turn flow)
- cancel: a convert cancelled over HTTP mid-port
"""
import json
import subprocess
import threading
import time

import httpx
import pytest
import yaml
from revolve_common import specs

from conftest import REPO, RUN, TASK, _recovery_conftest, in_image, load

import labels as L
import launch as D

CONTRACT = yaml.safe_load((REPO / "specs/robots/mobile_panda/contract.yaml").read_text())
ROBOT = yaml.safe_load((REPO / "specs/robots/mobile_panda/robot.yaml").read_text())
JOINTS = specs.load(REPO / "specs/robots/mobile_panda/robot.yaml").joints   # state / action derived
PROMPT = yaml.safe_load((REPO / f"specs/tasks/{TASK}/task.yaml").read_text())["prompt"]
SUMMARY = "meta/revolve_dataset.json"


def cancel_job(eps, seen):
    job = D.start(["convert", "--task", TASK, "--turn", "0", "--out", RUN / "cancel" / "dataset", *eps],
                  RUN / "cancel" / "job", linger_s=5.0)
    for _ in range(600):
        try:
            s = httpx.get(f"http://{job.ip()}:8000/status", timeout=2).json()
            if s["progress"].get("stage") == "port":
                seen["before"] = s
                seen["cancel"] = httpx.post(f"http://{job.ip()}:8000/cancel", timeout=5).status_code
                seen["t_cancel"] = time.time()
                break
        except Exception:
            pass
        time.sleep(0.5)
    seen["res"] = D.finish(job)
    seen["cancel_to_exit_s"] = round(time.time() - seen.get("t_cancel", time.time()), 1)


@pytest.fixture(scope="session")
def runs(episodes, numbers):
    RC = _recovery_conftest()
    demo, rec = episodes["demo"], episodes["recovery"]
    out, http, can = {}, {"status": [], "result": None}, {}
    t0 = time.time()

    def joint():
        job = D.start(["convert", "--task", TASK, "--turn", "0", "--out", RUN / "joint" / "turn_0" / "dataset", demo, rec],
                      RUN / "joint" / "turn_0" / "jobs" / "convert", linger_s=8.0)
        th = threading.Thread(target=RC.poll_http, args=(job, http), daemon=True)
        th.start()
        out["joint"] = D.finish(job)
        th.join(timeout=30)

    def loop():
        out["loop0"] = D.turn_dataset(TASK, RUN / "loop" / "turn_0", 0, [demo])
        out["loop1"] = D.turn_dataset(TASK, RUN / "loop" / "turn_1", 1, [rec], prev_dataset=RUN / "loop" / "turn_0" / "dataset")

    ths = [threading.Thread(target=f) for f in (joint, loop, lambda: cancel_job([demo, rec], can))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    numbers.update(fixtures_s=episodes["wall_s"], jobs_s=round(time.time() - t0, 1),
                   wall_s={k: v["wall_s"] for k, v in out.items()}, cancel=dict(can, res=None))
    return dict(out, http=http, cancel=can, demo=demo, rec=rec)


def test_image_is_deps_only():
    sha = subprocess.run(["git", "-C", str(REPO / "ThirdParty/rosetta"), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert (D.INSTALL / "BUILD_OK").read_text().strip() == sha
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", D.IMAGE, "bash", "-c",
                        "test ! -e /opt/rosetta/src && test ! -e /opt/rosetta/install && "
                        "! python3 -c 'import rosetta_ros' 2>/dev/null && python3 -c 'import lerobot; print(lerobot.__version__)'"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "0.6.1", r.stdout + r.stderr


def test_convert_over_http(runs, numbers):
    res, http = runs["joint"], runs["http"]
    assert res["exit"] == 0, res
    p = res["result"]["progress"]
    assert res["result"]["state"] == "succeeded" and p["stage"] == "done" and p["episodes_converted"] == 2 and p["problems"] == []
    assert http["status"] and http["result"]["state"] == "succeeded", http
    conv = [s["progress"].get("episodes_converted") for s in http["status"] if s["state"] == "running"]
    assert http.get("result_code_while_running") == 409 and conv, http
    numbers["http"] = dict(polls=len(http["status"]), converted_seen=sorted({c for c in conv if c is not None}))


def bag_images(ep):
    m = yaml.safe_load((ep / "bag" / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    return {t["topic_metadata"]["name"]: t["message_count"] for t in m["topics_with_message_count"]
            if t["topic_metadata"]["name"].startswith("/cameras/")}


def test_frames_match_bags(runs, numbers):
    s = load(RUN / "joint" / "turn_0" / "dataset" / SUMMARY)
    images = [bag_images(e) for e in (runs["demo"], runs["rec"])]
    cams = {k: v["channel"]["topic"] for k, v in CONTRACT["observations"].items() if k.startswith("observation.images.")}
    for n, im in zip(s["lengths"], images):
        assert sorted(im) == sorted(cams.values())
        assert all(c == n for c in im.values()), (n, im)          # one frame per camera image, none dropped
    assert s["episodes"] == 2 and s["frames"] == sum(s["lengths"])
    numbers["frames"] = dict(lengths=s["lengths"], bag_images=images)


def test_features_match_contract(runs):
    info = load(RUN / "joint" / "turn_0" / "dataset" / "meta" / "info.json")
    f = info["features"]
    for key, spec in CONTRACT["observations"].items():
        if key.startswith("observation.images."):
            assert f[key]["dtype"] == "video" and f[key]["shape"] == [480, 480, 3]
    # multi-source keys (Rosetta prefixes each source's names with its topic): the base twist in the robot frame
    # + joint positions, no world-frame base pose
    twist = ["linear.x", "linear.y", "angular.z"]
    assert f["observation.state"]["names"] == ["odom.twist.twist." + t for t in twist] + \
        ["joint_states.position." + j for j in JOINTS.policy_state[3:]]
    assert f["action"]["names"] == ["cmd_vel." + t for t in twist] + ["action.position." + j for j in JOINTS.policy_action[3:]]
    assert f["observation.state"]["shape"] == [12] and f["action"]["shape"] == [11]
    assert info["fps"] == CONTRACT["fps"] and info["robot_type"] == CONTRACT["robot_type"] and info["codebase_version"] == "v3.0"
    assert (RUN / "joint" / "turn_0" / "dataset" / "meta" / "rosetta_contract.yaml").read_text() == \
        (REPO / "specs/robots/mobile_panda/contract.yaml").read_text()


def test_prompt_and_labels(runs):
    s = load(RUN / "joint" / "turn_0" / "dataset" / SUMMARY)
    assert s["tasks"] == [PROMPT]
    demo, rec = (L.rel(p, REPO) for p in (runs["demo"], runs["rec"]))
    assert s["labels"] == [
        dict(source="demo", phase="", suffix=False, seed=0, turn=0, from_rollout="", episode_dir=demo),
        dict(source="recovery", phase="open", suffix=True, seed=0, turn=0, from_rollout=demo, episode_dir=rec)]


def test_loads_with_lerobot_and_values(runs, numbers):
    r = in_image("values", RUN / "joint" / "turn_0" / "dataset", runs["demo"], runs["rec"])
    assert r["total_frames"] == sum(e["length"] for e in r["episodes"])
    assert r["camera_keys"] == [k for k in CONTRACT["observations"] if k.startswith("observation.images.")]
    for e in r["episodes"]:
        assert e["length"] == e["meta_length"]
        assert e["state_action_max_abs_err"] == 0.0 and e["step_span_ok"], e      # every frame = a logged step
        assert e["state_dtype"] == "float64" and e["action_dtype"] == "float64"
        for fr in e["frames"].values():
            assert fr["task"] == PROMPT
            for key in r["camera_keys"]:
                assert fr[key]["shape"] == [3, 480, 480] and fr[key]["max"] > fr[key]["min"]
    assert [e["labels"]["source"] for e in r["episodes"]] == ["demo", "recovery"]
    numbers["values"] = [{k: e[k] for k in ("length", "first_step", "last_step", "state_action_max_abs_err")} for e in r["episodes"]]


def test_turn_merge(runs, numbers):
    for k in ("loop0", "loop1"):
        assert runs[k]["exit"] == 0, runs[k]
    t0, new, t1 = (load(RUN / "loop" / d / SUMMARY) for d in ("turn_0/dataset", "turn_1/dataset_new", "turn_1/dataset"))
    assert t1["kind"] == "merge" and t1["episodes"] == t0["episodes"] + new["episodes"] == 2
    assert t1["frames"] == t0["frames"] + new["frames"] and t1["lengths"] == t0["lengths"] + new["lengths"]
    assert [(lab["source"], lab["turn"]) for lab in t1["labels"]] == [("demo", 0), ("recovery", 1)]
    assert t1["labels"] == t0["labels"] + new["labels"] and t1["tasks"] == [PROMPT]
    # merged = the joint convert, frame for frame (all data columns; LeRobot's global index columns aside)
    c = in_image("compare", RUN / "joint" / "turn_0" / "dataset", RUN / "loop" / "turn_1" / "dataset")
    assert c["identical"], c["diffs"]
    r = in_image("values", RUN / "loop" / "turn_1" / "dataset", runs["demo"], runs["rec"])
    assert [e["labels"]["turn"] for e in r["episodes"]] == [0, 1]
    assert all(e["state_action_max_abs_err"] == 0.0 for e in r["episodes"])
    numbers["merge"] = dict(episodes=t1["episodes"], frames=t1["frames"], wall_s=runs["loop1"]["wall_s"])


def test_resume_keeps_done_dataset(runs):
    t = time.time()
    res = D.convert(TASK, [runs["demo"]], RUN / "loop" / "turn_0" / "dataset", 0)
    assert res["exit"] == 0 and res["result"]["progress"].get("resumed") is True, res
    assert time.time() - t < 60


def test_cancel(runs, numbers):
    c = runs["cancel"]
    assert c.get("cancel") == 200, c
    assert c["res"]["result"]["state"] == "cancelled", c["res"]
    assert not (RUN / "cancel" / "dataset").exists()
    assert c["cancel_to_exit_s"] < 30
    numbers["cancel_to_exit_s"] = c["cancel_to_exit_s"]
