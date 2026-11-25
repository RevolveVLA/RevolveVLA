"""Real conversions in the dataset image (DATASET-v2.md sections 4, 7, 8, 11, 14): simulator episodes recorded with
tests/bags.py (fixture manifests listing them as passed), converted with launch.run, loaded with LeRobotDataset inside
the image, split, a late action stream recorded and warned about, resumed, and run under the HTTP job wrapper (status
poll, 409, cancel). Session fixtures: real_fixtures.py. Outputs and numbers.json: runs/dataset_tests/<timestamp>/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from conftest import (ROBOT_TASK, contract_action_topics, contract_path, convert_argv, last_json_line, prompt, sha256,
                      tree_mtimes)
from image import job_ip
from real_fixtures import (CAMS, FPS, HAND, ROBOTS, STEP_NS, TOLERANCE, expected_entries, floor_frames, load_in_image,
                           record_of, rewritten_run, run_cli, tail)

pytest_plugins = ["real_fixtures"]

TERMINAL = ("succeeded", "failed", "cancelled")
CONVERT_CHECKS = {"port", "frames", "features", "task", "labels", "episodes"}


def _names(arm_ctl, hand_ctl, arm, hand):
    base_s = [f"omni_wheel_drive_controller.twist.twist.{a}" for a in ("linear.x", "linear.y", "angular.z")]
    base_a = [f"omni_wheel_drive_controller.twist.{a}" for a in ("linear.x", "linear.y", "angular.z")]
    return {"observation.state": base_s + [f"joint_states.position.{j}" for j in (*arm, hand)],
            "action": base_a + [f"{arm_ctl}.position.{j}" for j in arm] + [f"{hand_ctl}.position.{hand}"]}


SO101_ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# DATASET-v2 5.5: feature names in order
FEATURE_NAMES = {
    "mobile_panda": _names("panda_arm_controller", "panda_hand_controller",
                           [f"panda_joint{i}" for i in range(1, 8)], "panda_finger_joint1"),
    "mobile_so101": _names("arm_controller", "gripper_controller", SO101_ARM, "gripper"),
    "mobile_so101_xl": _names("arm_controller", "gripper_controller", SO101_ARM, "gripper"),
}
WIDTH = {"mobile_panda": 11, "mobile_so101": 9, "mobile_so101_xl": 9}


# ------------------------------------------------------------------ per robot: convert

@pytest.mark.parametrize("robot", ROBOTS)
def test_convert_succeeds(converted, robot):
    # DATASET-v2 7/8.4: exit 0, the last stdout line the result JSON {dataset, episodes, frames, problems: []}
    c = converted(robot)
    assert c.rc == 0, c.why()
    res = last_json_line(c.text)
    want = expected_entries(c.manifest)
    assert res["dataset"] == str(c.out)
    assert res["episodes"] == len(want) and res["problems"] == []
    assert res["frames"] == c.info()["total_frames"] == sum(p["frames"] for p in c.summary()["per_episode"])
    assert not res.get("resumed"), "a first conversion is not a resume"


@pytest.mark.parametrize("robot", ROBOTS)
def test_convert_layout(converted, robot):
    # DATASET-v2 8.1: LeRobot v3.0 layout, info.json fps 30 / v3.0 / robot_type, the contract copied byte for byte,
    # no <out>.partial left after success
    c = converted(robot)
    assert c.rc == 0, c.why()
    info = c.info()
    assert info["fps"] == FPS and info["codebase_version"] == "v3.0" and info["robot_type"] == robot
    assert info["total_episodes"] == len(expected_entries(c.manifest))
    assert (c.out / "meta" / "rosetta_contract.yaml").read_bytes() == contract_path(robot).read_bytes()
    assert (c.out / "meta" / "stats.json").is_file()
    assert (c.out / "meta" / "tasks.parquet").is_file()
    assert list((c.out / "meta" / "episodes").rglob("*.parquet"))
    assert list((c.out / "data").glob("chunk-*/file-*.parquet"))
    for cam in CAMS:
        assert list((c.out / "videos" / cam).glob("chunk-*/file-*.mp4")), cam
    assert not c.out.with_name(c.out.name + ".partial").exists()


@pytest.mark.parametrize("robot", ROBOTS)
def test_convert_summary_and_frames(converted, robot):
    # DATASET-v2 8.2/8.3 + decisions: revolve_dataset.json holds kind, inputs (the resume key: the manifests, their
    # and the contract's sha256 and the task prompt, so a changed prompt is not a resume), contract, prompt, features,
    # exactly the convert checks, all ok; per episode labels in manifest order, frames within 6 of
    # floor(duration_s x 30) and == bag_ticks - warmup
    c = converted(robot)
    assert c.rc == 0, c.why()
    s = c.summary()
    want = expected_entries(c.manifest)
    assert s["kind"] == "convert"
    assert s["problems"] == []
    assert set(s["checks"]) == CONVERT_CHECKS and all(v == "ok" for v in s["checks"].values()), s["checks"]
    inputs = json.dumps(s["inputs"])
    assert str(c.manifest) in inputs and sha256(c.manifest) in inputs and sha256(contract_path(robot)) in inputs
    assert json.dumps(prompt(ROBOT_TASK[robot])) in inputs, s["inputs"]
    assert s["prompt"] == prompt(ROBOT_TASK[robot])
    assert s["frame_tolerance"] == TOLERANCE
    assert s["episodes"] == len(want) and s["frames"] == sum(p["frames"] for p in s["per_episode"])
    feats = {k: list(v) for k, v in s["features"].items()}
    assert feats == {**{k: [480, 480, 3] for k in CAMS}, "observation.state": [WIDTH[robot]],
                     "action": [WIDTH[robot]]}
    assert len(s["per_episode"]) == len(want)
    for p, (e, labels) in zip(s["per_episode"], want, strict=True):
        assert p["labels"] == labels
        assert Path(p["bag"]).name == Path(e["bag"]).name and Path(p["result"]).name == Path(e["result"]).name
        dur = json.loads((c.manifest.parent / e["result"]).read_text())["duration_s"]
        assert p["duration_s"] == dur
        assert abs(p["frames"] - floor_frames(dur)) <= TOLERANCE, p
        assert p["offset"] == p["frames"] - floor_frames(dur), p
        assert p["frames"] == p["bag_ticks"] - p["warmup_ticks"], p
        assert p["warmup_ticks"] >= 0 and p["frames"] > 0, p

    # late action streams (Rosetta zero-fills an action until its stream's first message): every episode records
    # action_start_frames for every action topic of the contract; each late topic is one `frames:` warning, in the
    # record, on stderr and in the result line, and the run still exits 0 with no problems
    topics = contract_action_topics(robot)
    for p in s["per_episode"]:
        asf = p["action_start_frames"]
        assert set(asf) == set(topics), asf
        assert all(type(n) is int and n >= 0 for n in asf.values()), asf
    late = sum(1 for p in s["per_episode"] for n in p["action_start_frames"].values() if n > 0)
    assert isinstance(s["warnings"], list) and len(s["warnings"]) == late, s["warnings"]
    assert all(w.startswith("frames:") and " starts " in w for w in s["warnings"]), s["warnings"]
    res = last_json_line(c.text)
    if s["warnings"]:
        assert res.get("warnings") == s["warnings"], res
        err = Path(c.err).read_text(errors="replace")
        assert all(w in err for w in s["warnings"]), "every warning is printed to stderr"
    else:
        assert not res.get("warnings"), res


@pytest.mark.parametrize("robot", ROBOTS)
def test_feature_names(converted, robot):
    # DATASET-v2 5.5: observation.state / action names in order (Rosetta's topic-segment prefixes), float64; images
    # video 480 x 480 x 3; the bookkeeping keys present
    c = converted(robot)
    assert c.rc == 0, c.why()
    f = c.info()["features"]
    for key, names in FEATURE_NAMES[robot].items():
        got = f[key]["names"]
        if isinstance(got, dict) and len(got) == 1:          # LeRobot may nest the list under one axis name
            got = next(iter(got.values()))
        assert list(got) == names, (key, got)
        assert f[key]["dtype"] == "float64" and list(f[key]["shape"]) == [WIDTH[robot]]
    for cam in CAMS:
        assert f[cam]["dtype"] == "video" and list(f[cam]["shape"]) == [480, 480, 3]
    for k in ("timestamp", "frame_index", "episode_index", "index", "task_index", "is_first", "is_last",
              "is_terminal"):
        assert k in f, k
    extra = set(f) - set(CAMS) - {"observation.state", "action", "timestamp", "frame_index", "episode_index",
                                  "index", "task_index", "is_first", "is_last", "is_terminal"}
    assert not extra, f"features beyond the contract (DS4: no ground truth in observation.*): {extra}"


@pytest.mark.parametrize("robot", ROBOTS)
def test_load_with_lerobot(ready, converted, robot):
    # DATASET-v2 8.1/14: LeRobotDataset loads it in the image and decodes its last frame; images 3 x 480 x 480 and
    # not constant; every frame's task == task.yaml prompt; labels/* columns in meta.episodes, in order
    c = converted(robot)
    assert c.rc == 0, c.why()
    s = c.summary()
    r = load_in_image(ready, c.out)
    want = expected_entries(c.manifest)
    p = prompt(ROBOT_TASK[robot])
    assert r["len"] == s["frames"] and r["num_episodes"] == len(want) and r["fps"] == FPS
    assert r["first_task"] == p and r["last_task"] == p
    assert r["task_indices"] == [0] and r["tasks"] == {"0": p}
    for k, v in r["images"].items():
        assert v["shape"] == [3, 480, 480], (k, v)
        assert v["std"] > 0, f"{k} is constant"
    assert r["state"] == [WIDTH[robot]] and r["action"] == [WIDTH[robot]]
    assert all(f"labels/{n}" in r["columns"] for n in want[0][1]), r["columns"]
    assert r["labels"] == [lab for _, lab in want]
    assert r["lengths"] == [pe["frames"] for pe in s["per_episode"]]


def test_split_train_and_val(converted):
    # DATASET-v2 6 --split: train keeps the 2 train episodes, val the 1 val episode, labels split accordingly
    for split, n in (("train", 2), ("val", 1)):
        c = converted("mobile_panda", split)
        assert c.rc == 0, c.why()
        s = c.summary()
        assert s["episodes"] == n == len(expected_entries(c.manifest, split))
        assert [p["labels"] for p in s["per_episode"]] == [lab for _, lab in expected_entries(c.manifest, split)]
        assert all(p["labels"]["split"] == split for p in s["per_episode"])
        assert split in json.dumps(s["inputs"])


# ------------------------------------------------------------------ a late action stream

def test_late_hand_stream_warns_and_exits_0(ready, fixtures, outdir):
    # the hand stream's messages of the first 3 s dropped (Rosetta zero-fills those frames and logs nothing) -> exit
    # 0, no problems, action_start_frames[hand] == N computed here on the porter's grid (ticks i >= warmup,
    # start + i * step < the first hand receive time), and a warning "frames: <name>: action <hand> starts N frames
    # after the first frame ..." in the record, the result line and stderr
    root = outdir / "late_hand"
    man, r = rewritten_run(ready, fixtures, root, "late:3")
    assert r["dropped"] > 0 and r["first_hand_ns"], r
    out = root / "ds"
    rc, text, _, err = run_cli(ready, convert_argv([man], out), outdir / "logs" / "late_hand")
    assert rc == 0, tail(text)
    res = last_json_line(text)
    assert res["problems"] == [], res
    s = record_of(out)
    (pe,) = s["per_episode"]
    meta = yaml.safe_load((root / "episode_0_bag" / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    start = int(meta["starting_time"]["nanoseconds_since_epoch"])
    first, warmup, ticks = int(r["first_hand_ns"]), int(pe["warmup_ticks"]), int(pe["bag_ticks"])
    want = sum(1 for i in range(warmup, ticks) if start + i * STEP_NS < first)
    assert want >= 60, (want, start, first, warmup)          # ~3 s late at 30 fps, minus the warmup
    assert pe["action_start_frames"][HAND] == want, (pe["action_start_frames"], want)
    tag = f": action {HAND} starts {want} frames after the first frame"
    hits = [w for w in s["warnings"] if w.startswith("frames:") and tag in w]
    assert hits, s["warnings"]
    assert hits[0] in res.get("warnings", []), res
    assert hits[0] in Path(err).read_text(errors="replace")


# ------------------------------------------------------------------ resume

def test_resume_same_inputs(ready, converted, outdir, numbers):
    # DATASET-v2 8.5: same inputs + same out -> exit 0, "resumed": true, fast, the dataset untouched (same mtimes)
    c = converted("mobile_so101")
    assert c.rc == 0, c.why()
    before = tree_mtimes(c.out)
    rc, text, wall, _ = run_cli(ready, c.argv, outdir / "logs" / "resume_same")
    numbers("resume/mobile_so101", {"rc": rc, "wall_s": wall})
    assert rc == 0, tail(text)
    res = last_json_line(text)
    assert res.get("resumed") is True and res["problems"] == [], res
    assert res["episodes"] == c.summary()["episodes"]
    assert wall < 60, f"resume took {wall} s"
    assert tree_mtimes(c.out) == before


def test_changed_manifest_rebuilt(ready, fixtures, outdir, numbers):
    # DATASET-v2 8.5: the same manifest path with changed content is not a resume: rebuilt, with the new inputs
    src = fixtures("mobile_so101")
    entries = [dict(e, bag=str(src.parent / e["bag"]), result=str(src.parent / e["result"]))
               for e, _ in expected_entries(src)]
    root = outdir / "resume_changed"
    root.mkdir()
    man = root / "manifest.json"
    doc = {"task": "open_drawer", "robot": "mobile_so101", "episodes": entries[:1]}
    man.write_text(json.dumps(doc, indent=1))
    out = root / "ds"
    argv = convert_argv([man], out)
    rc, text, wall, _ = run_cli(ready, argv, outdir / "logs" / "resume_changed_1")
    assert rc == 0, tail(text)
    assert last_json_line(text)["episodes"] == 1
    rc, text, _, _ = run_cli(ready, argv, outdir / "logs" / "resume_changed_2")
    assert rc == 0 and last_json_line(text).get("resumed") is True, tail(text)
    doc["episodes"] = entries[:2]
    man.write_text(json.dumps(doc, indent=1))
    rc, text, wall2, _ = run_cli(ready, argv, outdir / "logs" / "resume_changed_3")
    numbers("resume_changed/mobile_so101", {"first_wall_s": wall, "rebuild_wall_s": wall2})
    assert rc == 0, tail(text)
    res = last_json_line(text)
    assert res.get("resumed") is not True and res["episodes"] == 2, res
    s = record_of(out)
    assert sha256(man) in json.dumps(s["inputs"]) and s["episodes"] == 2
    assert not out.with_name(out.name + ".partial").exists()


# ------------------------------------------------------------------ the HTTP job wrapper

def test_wrapper_convert(ready, fixtures, outdir, numbers):
    # launch.convert (start/wait under revolve_common.wrapper): result.json state succeeded, progress stage done with
    # dataset / episodes / frames / problems; default job dir <out parent>/jobs/dataset_convert_<name>
    m = fixtures("mobile_panda")
    out = outdir / "wrapper_val"
    r = ready.convert([str(m)], str(out), turn=0, split="val")
    numbers("wrapper/convert_val", {"exit": r["exit"], "wall_s": r["wall_s"]})
    assert r["exit"] == 0, r
    res = r["result"]
    assert res["state"] == "succeeded" and res["returncode"] == 0, res
    prog = res["progress"]
    assert prog["stage"] == "done" and prog["problems"] == [] and prog["episodes"] == 1 and prog["frames"] > 0, prog
    assert str(prog["dataset"]) == str(out)
    jd = out.parent / "jobs" / f"dataset_convert_{out.name}"
    assert Path(r["dir"]).resolve() == jd.resolve()
    for f in ("job.json", "status.json", "result.json", "stdout.log"):
        assert (jd / f).is_file(), f
    assert json.loads((jd / "result.json").read_text())["state"] == "succeeded"
    assert (out / "meta" / "revolve_dataset.json").is_file()


def _poll(job, until, deadline_s=1800):
    """GET /status (and /result while running) at the job container's :8000 until until(status) -> (status, seen)."""
    import httpx
    seen = {"running": 0, "result_409": 0, "stages": []}
    t_end = time.time() + deadline_s
    status = None
    while time.time() < t_end:
        try:
            base = f"http://{job_ip(job)}:8000"
            status = httpx.get(f"{base}/status", timeout=5).json()
        except Exception:                  # not up yet, or gone after the linger
            if status and status.get("state") in TERMINAL:
                break
            time.sleep(0.5)
            continue
        stage = status.get("progress", {}).get("stage")
        if stage and (not seen["stages"] or seen["stages"][-1] != stage):
            seen["stages"].append(stage)
        if status.get("state") == "running":
            seen["running"] += 1
            try:
                if httpx.get(f"{base}/result", timeout=5).status_code == 409:
                    seen["result_409"] += 1
            except Exception:
                pass
        if until(status):
            break
        time.sleep(0.5)
    return status, seen


def test_wrapper_status_and_409(ready, fixtures, outdir, numbers):
    # DATASET-v2 7: the HTTP wrapper is the outer contract: GET /status while running, GET /result 409 until finished
    m = fixtures("mobile_so101")
    out = outdir / "wrapper_poll"
    jd = outdir / "jobs" / "wrapper_poll"
    job = ready.start(convert_argv([m], out), jd)
    assert job.container and Path(job.dir).resolve() == jd.resolve()
    status, seen = _poll(job, lambda s: s.get("state") in TERMINAL)
    r = ready.wait(job, 1800)
    numbers("wrapper/poll", {"exit": r["exit"], "wall_s": r["wall_s"], "stages": seen["stages"]})
    assert seen["running"] >= 1, f"never saw the job running: {status}"
    assert seen["result_409"] >= 1, "GET /result never answered 409 while running"
    assert r["exit"] == 0 and r["result"]["state"] == "succeeded", r
    assert r["result"]["progress"]["stage"] == "done"
    assert "port" in seen["stages"] or "checks" in seen["stages"], seen


def test_wrapper_cancel(ready, fixtures, outdir, numbers):
    # DATASET-v2 7: POST /cancel during the port -> result state cancelled, nothing at out
    import httpx
    m = fixtures("mobile_panda")
    out = outdir / "wrapper_cancel"
    jd = outdir / "jobs" / "wrapper_cancel"
    job = ready.start(convert_argv([m], out), jd)
    status, seen = _poll(job, lambda s: s.get("progress", {}).get("stage") == "port" or s.get("state") in TERMINAL)
    assert status and status.get("state") == "running" and status["progress"].get("stage") == "port", \
        f"never saw stage port while running: {status}, {seen}"
    resp = httpx.post(f"http://{job_ip(job)}:8000/cancel", timeout=10)
    assert resp.status_code == 200 and resp.json().get("cancelled") is True, resp.text
    r = ready.wait(job, 600)
    numbers("wrapper/cancel", {"exit": r["exit"], "wall_s": r["wall_s"]})
    assert r["result"]["state"] == "cancelled", r
    assert r["exit"] != 0
    assert not out.exists(), "nothing at <out> after a cancel"
