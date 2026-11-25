"""Session fixtures and helpers of the real-conversion suite (test_convert.py, test_merge.py, test_fail_closed.py),
loaded by each of them with `pytest_plugins = ["real_fixtures"]`: registered once, so every conversion and merge here
runs once per session whichever modules use it.

  ready              the launch module, once the image and the Rosetta build exist (fails, never skips)
  outdir, numbers    runs/dataset_tests/<timestamp>/ and its numbers.json
  fixtures(robot)    the robot's fixture manifest under runs/test_cache/dataset/<robot>/ (recorded when missing)
  converted(robot, split=None)    one convert of that manifest per (robot, split) -> Conv
  merged             mobile_panda's train and val conversions merged -> (Conv, train Conv, val Conv)

Needs `./revolve build dataset` (the image and ThirdParty/rosetta/install/BUILD_OK); the first run also needs
`./revolve build simulator`, the scene assets and a GPU to record the fixtures (reused after)."""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

import bags
from conftest import PREFIXES, REPO, ROBOT_TASK, convert_argv, last_json_line, progress_lines
from image import in_image

CACHE = REPO / "runs" / "test_cache" / "dataset"
# per robot: (split, seed, episodes) recorded into one root / one fixture manifest
PLAN = {"mobile_panda": [("train", 0, 2), ("val", 1, 1)],
        "mobile_so101": [("train", 0, 2)],
        "mobile_so101_xl": [("train", 0, 2)]}
ROBOTS = list(PLAN)
FPS, TOLERANCE = 30, 6
STEP_NS = int(1e9 / FPS)               # Rosetta's step_ns(30) (rosetta/frames/resample.py)
REPO_ID = "revolve/dataset"            # any fixed local id: the dataset is loaded from root, offline
CAMS = ("observation.images.wrist", "observation.images.overhead_left", "observation.images.overhead_right")
HAND = "/panda_hand_controller/joint_trajectory"


def floor_frames(duration_s, fps=FPS):
    """floor(duration_s x fps) of the decimal value as written (4.1 x 30 = 123)."""
    return math.floor(Decimal(repr(float(duration_s))) * fps)


def tail(text, n=40):
    return "\n".join(text.splitlines()[-n:])


# ------------------------------------------------------------------ session: prerequisites, output dir, numbers

@pytest.fixture(scope="session")
def ready(launch):
    """The launch module, once the image and the Rosetta build exist (fail, not skip: these tests are the proof)."""
    if not (Path(launch.INSTALL) / "BUILD_OK").is_file():
        pytest.fail(f"{launch.INSTALL}/BUILD_OK is missing: run `./revolve build dataset` first")
    if shutil.which("docker") is None:
        pytest.fail("docker is not installed: the real dataset tests run in the dataset image")
    r = subprocess.run(["docker", "image", "inspect", launch.IMAGE], capture_output=True, text=True)
    if r.returncode:
        pytest.fail(f"the dataset image {launch.IMAGE} is missing: run `./revolve build dataset` first")
    return launch


@pytest.fixture(scope="session")
def outdir():
    d = REPO / "runs" / "dataset_tests" / time.strftime("%Y%m%d_%H%M%S")
    (d / "logs").mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def numbers(outdir):
    """note(key, value): numbers.json in the output dir, rewritten on every note (survives a crash)."""
    data = {}

    def note(key, value):
        data[key] = value
        (outdir / "numbers.json").write_text(json.dumps(data, indent=1, default=str))
    return note


# ------------------------------------------------------------------ session: recorded fixtures

def _part_ok(root, doc, split, seed, n):
    es = [e for e in doc.get("episodes", []) if str(e.get("scene", "")).startswith(f"{split}/scene_{seed}/")]
    return len(es) == n and all((root / e["bag"] / "metadata.yaml").is_file() and (root / e["result"]).is_file()
                                for e in es)


@pytest.fixture(scope="session")
def fixtures(ready, numbers):
    """fixtures(robot) -> the fixture manifest under runs/test_cache/dataset/<robot>/ (recorded when missing)."""
    made = {}

    def get(robot):
        if robot in made:
            return made[robot]
        root, task = CACHE / robot, ROBOT_TASK[robot]
        m = root / bags.FIXTURE
        for split, seed, n in PLAN[robot]:
            doc = json.loads(m.read_text()) if m.is_file() else {}
            if _part_ok(root, doc, split, seed, n):
                continue
            r = subprocess.run(["docker", "image", "inspect", bags.SIM_IMAGE], capture_output=True)
            if r.returncode:
                pytest.fail(f"no cached fixture for {robot} {split}/scene_{seed} and no simulator image "
                            f"{bags.SIM_IMAGE} to record it: run `./revolve build simulator` (and `./revolve assets`)")
            t0 = time.time()
            try:
                bags.record(root, robot, task, n, split=split, seed=seed)
            except Exception as e:                       # a broken recording is a fixture failure, said clearly
                pytest.fail(f"recording the {robot} fixture ({split}, seed {seed}, {n} episodes) failed: {e}")
            numbers(f"record/{robot}/{split}_seed{seed}", {"episodes": n, "wall_s": round(time.time() - t0, 1)})
        doc = json.loads(m.read_text())
        for split, seed, n in PLAN[robot]:
            assert _part_ok(root, doc, split, seed, n), f"fixture {m} incomplete for {split}/scene_{seed}"
        made[robot] = m
        return m
    return get


def expected_entries(manifest, split=None, turn=0):
    """[(entry, labels)] of the passed entries in manifest order (DATASET-v2 6), computed independently."""
    doc = json.loads(Path(manifest).read_text())
    out = []
    for e in doc["episodes"]:
        if e.get("passed") is not True:
            continue
        sp = e.get("split", e["scene"].split("/")[0])
        if split is not None and sp != split:
            continue
        out.append((e, {"source": "demo", "seed": e["seed"], "split": sp, "scene": e["scene"], "episode": e["episode"],
                        "expert": e["datagen"]["expert"], "attempt": e["datagen"]["attempt"], "turn": turn,
                        "manifest": str(manifest)}))
    return out


# ------------------------------------------------------------------ running the CLI (launch.run) with its output

def run_cli(launch, argv, log: Path):
    """launch.run(argv) with fd 1 / fd 2 (the docker client's output) and sys.stdout / sys.stderr in files
    -> (rc, stdout text, wall_s, stderr path)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    out_p, err_p = log.with_name(log.name + ".stdout"), log.with_name(log.name + ".stderr")
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    t0 = time.time()
    with open(out_p, "w", buffering=1) as fo, open(err_p, "w", buffering=1) as fe:
        try:
            os.dup2(fo.fileno(), 1)
            os.dup2(fe.fileno(), 2)
            with contextlib.redirect_stdout(fo), contextlib.redirect_stderr(fe):
                rc = launch.run(list(map(str, argv)))
                sys.stdout.flush()
                sys.stderr.flush()
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
    return rc, out_p.read_text(errors="replace"), round(time.time() - t0, 1), err_p


def record_of(out: Path) -> dict:
    return json.loads((Path(out) / "meta" / "revolve_dataset.json").read_text())


@dataclass
class Conv:
    robot: str
    split: str | None
    manifest: Path
    out: Path
    argv: list
    rc: int
    text: str
    wall_s: float
    err: Path

    def why(self):
        return f"{self.argv}: rc {self.rc}\n{tail(self.text)}\nstderr: {self.err}"

    def summary(self):
        return record_of(self.out)

    def info(self):
        return json.loads((self.out / "meta" / "info.json").read_text())


def note_episodes(numbers, key, c: Conv):
    pe = []
    if c.rc == 0:
        for p in c.summary().get("per_episode", []):
            pe.append({k: p.get(k) for k in ("manifest_entry", "duration_s", "frames", "offset", "bag_ticks",
                                             "warmup_ticks")}
                      | {"floor_duration_x30": floor_frames(p["duration_s"]) if "duration_s" in p else None})
    numbers(key, {"rc": c.rc, "wall_s": c.wall_s, "out": str(c.out), "per_episode": pe})


@pytest.fixture(scope="session")
def converted(ready, fixtures, outdir, numbers):
    """converted(robot, split=None) -> Conv: one convert of the robot's fixture manifest per (robot, split)."""
    done = {}

    def get(robot, split=None):
        if (robot, split) not in done:
            m = fixtures(robot)
            out = outdir / (robot + (f"_{split}" if split else ""))
            argv = convert_argv([m], out, *(["--split", split] if split else []))
            rc, text, wall, err = run_cli(ready, argv, outdir / "logs" / f"convert_{out.name}")
            c = Conv(robot, split, m, out, argv, rc, text, wall, err)
            note_episodes(numbers, f"convert/{out.name}", c)
            done[(robot, split)] = c
        return done[(robot, split)]
    return get


@pytest.fixture(scope="session")
def merged(ready, converted, outdir, numbers):
    """mobile_panda's train and val datasets merged in that order -> (merge Conv, train Conv, val Conv)."""
    t, v = converted("mobile_panda", "train"), converted("mobile_panda", "val")
    assert t.rc == 0 and v.rc == 0, (t.why(), v.why())
    out = outdir / "mobile_panda_merged"
    argv = ["merge", "--out", str(out), str(t.out), str(v.out)]
    rc, text, wall, err = run_cli(ready, argv, outdir / "logs" / "merge")
    c = Conv("mobile_panda", None, t.manifest, out, argv, rc, text, wall, err)
    numbers("merge/mobile_panda_merged", {"rc": rc, "wall_s": wall})
    return c, t, v


def failed_closed(rc, text, out, prefix):
    """DATASET-v2 7 / 8.4: exit 1, the result JSON's problems (each with a check's prefix, one with `prefix`), no
    PROGRESS `done`, nothing at <out>."""
    assert rc == 1, tail(text)
    res = last_json_line(text)
    assert res.get("problems"), res
    assert all(isinstance(p, str) and p.startswith((*PREFIXES, "out:")) for p in res["problems"]), res
    assert any(p.startswith(prefix) for p in res["problems"]), res
    assert all(pr.get("stage") != "done" for pr in progress_lines(text)), text
    assert not Path(out).exists(), f"{out} exists after a failure"
    return res["problems"]


def copies(converted, root: Path, *which):
    """Copies of finished mobile_panda datasets (which: splits) under root, for tests that edit or risk their inputs."""
    out = []
    for split in which:
        c = converted("mobile_panda", split)
        assert c.rc == 0, c.why()
        shutil.copytree(c.out, root / split)
        out.append(root / split)
    return out


# ------------------------------------------------------------------ commands in the image

def image_python(launch, code, *args, mounts_for=()):
    """Run python3 -c code in the dataset image -> the RESULT JSON it prints."""
    r = in_image(launch, ["python3", "-c", code, *map(str, args)], mounts_for=list(map(str, mounts_for)))
    assert r.returncode == 0, f"in_image failed (rc {r.returncode}):\n{tail(r.stdout)}\n{tail(r.stderr)}"
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"no RESULT line:\n{tail(r.stdout)}\n{tail(r.stderr)}"
    return json.loads(lines[-1][len("RESULT "):])


LOAD = r'''
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo_id, root = sys.argv[1], sys.argv[2]
ds = LeRobotDataset(repo_id, root=root)
n = len(ds)
frames = {"first": ds[0], "last": ds[n - 1]}
images = {}
for which, fr in frames.items():
    for k in ds.meta.camera_keys:
        v = fr[k]
        images[f"{which}/{k}"] = {"shape": list(v.shape), "std": float(v.float().std())}
per_frame = sorted({int(x) for x in ds.hf_dataset["task_index"]})
tasks = {}
for name, row in ds.meta.tasks.iterrows():
    tasks[str(int(row["task_index"]))] = name
eps = ds.meta.episodes
cols = list(eps.column_names)
labels = [{c[len("labels/"):]: eps[i][c] for c in cols if c.startswith("labels/")} for i in range(len(eps))]
lengths = [int(eps[i]["length"]) for i in range(len(eps))]
print("RESULT " + json.dumps({
    "len": n, "num_episodes": ds.num_episodes, "fps": ds.fps,
    "first_task": frames["first"]["task"], "last_task": frames["last"]["task"],
    "task_indices": per_frame, "tasks": tasks, "images": images, "labels": labels, "lengths": lengths,
    "state": list(frames["last"]["observation.state"].shape), "action": list(frames["last"]["action"].shape),
    "columns": cols}, default=lambda o: f"<{type(o).__name__}> {o}"))
'''


def load_in_image(launch, out: Path):
    # the dataset given as a merge-style input: mounted read-only at its own path (it is under runs/ anyway)
    return image_python(launch, LOAD, REPO_ID, out, mounts_for=["merge", "--out", str(out) + ".loadcheck", str(out)])


# rosbag2 rewrite of one bag in the image, by mode: "tamper" sets every hand joint_trajectory message's joint_names to
# a joint the contract does not select; "plain" copies unchanged; "late:<s>" drops the hand messages of the first <s>
# seconds after the bag start
REWRITE = r'''
import json, shutil, sys
from pathlib import Path
import rosbag2_py, yaml
from rclpy.serialization import deserialize_message, serialize_message
from trajectory_msgs.msg import JointTrajectory
src, out, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
HAND = "/panda_hand_controller/joint_trajectory"
meta = yaml.safe_load((src / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
custom = meta.get("custom_data") or {}
dst = out / "episode_0_bag"
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=str(src), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
start = r.get_metadata().starting_time.nanoseconds
w = rosbag2_py.SequentialWriter()
w.open(rosbag2_py.StorageOptions(uri=str(dst), storage_id="mcap", custom_data=custom),
       rosbag2_py.ConverterOptions("cdr", "cdr"))
for t in r.get_all_topics_and_types():
    w.create_topic(t)
changed = dropped = 0
first_hand = None
cut = start + int(float(mode.split(":")[1]) * 1e9) if mode.startswith("late:") else None
while r.has_next():
    topic, data, ts = r.read_next()
    if topic == HAND:
        if mode == "tamper":
            m = deserialize_message(data, JointTrajectory); m.joint_names = ["panda_finger_joint2"]
            data = serialize_message(m)
            changed += 1
        elif cut is not None and ts < cut:
            dropped += 1
            continue
        if first_hand is None:
            first_hand = ts
    w.write(topic, data, ts)
del w; r.close()
print("RESULT " + json.dumps({"changed": changed, "dropped": dropped, "first_hand_ns": first_hand,
                             "src_start_ns": start}))
'''


def first_entry(fixtures, robot="mobile_panda", n=1):
    """The robot's fixture manifest and its first n passed entries."""
    m = fixtures(robot)
    es = [e for e, _ in expected_entries(m)][:n]
    assert len(es) == n, f"the {robot} fixture has fewer than {n} passed entries"
    return m, es


def write_run_manifest(root: Path, m: Path, es: list) -> Path:
    """root/manifest.json listing entry i (of manifest m) as `episode_<i>_bag` and `episode_<i>.json` under root; the
    results copied, the bags left to the caller."""
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, e in enumerate(es):
        shutil.copy(m.parent / e["result"], root / f"episode_{i}.json")
        entries.append(dict(e, bag=f"episode_{i}_bag", result=f"episode_{i}.json"))
    doc = json.loads(m.read_text())
    man = root / "manifest.json"
    man.write_text(json.dumps({"task": doc["task"], "robot": doc["robot"], "episodes": entries}, indent=1))
    return man


def link_bag(src: Path, dst: Path):
    """A bag dir whose files are symlinks to the cached bag's (cheap; the cache is never written)."""
    dst.mkdir(parents=True)
    for f in src.iterdir():
        (dst / f.name).symlink_to(f.resolve())


def rewritten_run(ready, fixtures, root: Path, mode: str):
    """A one-episode producer run at root whose bag is the first fixture bag rewritten by REWRITE in `mode` ->
    (manifest, REWRITE's RESULT)."""
    m, es = first_entry(fixtures)
    root.mkdir(parents=True, exist_ok=True)
    src = m.parent / es[0]["bag"]
    r = image_python(ready, REWRITE, src, root, mode,
                     mounts_for=["merge", "--out", str(root / "unused"), str(src.resolve())])
    return write_run_manifest(root, m, es), r


def light_snapshot(root: Path) -> dict:
    """Every entry under root: dirs, files with size and mtime (bags are ~135 MB: no hashing)."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = ("link", str(p.readlink()))
        elif p.is_file():
            st = p.stat()
            out[rel] = ("file", st.st_size, st.st_mtime_ns)
        else:
            out[rel] = ("dir",)
    return out
