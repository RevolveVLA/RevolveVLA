"""Dataset v2 tests (components/dataset/DATASET-v2.md).

  *_unit.py       host only, fast, no docker: labels, contract, manifest, checks, ticks, port's log reader, the CLI's
                  out and input checks, launch.docker_cmd / start / wait. Run with `-k unit`.
  test_convert.py, test_merge.py, test_fail_closed.py
                  real conversions of simulator-recorded bags in the dataset image (needs `./revolve build dataset`,
                  the simulator image and a GPU for the first run's recordings, cached under
                  runs/test_cache/dataset/<robot>/). Their session fixtures live in real_fixtures.py.

Helpers here build synthetic producer runs in tmp_path: episode bags with only a rosbag2 `metadata.yaml`
(topics_with_message_count, the real contract topics of specs/robots/<robot>/contract.yaml read as plain YAML),
evaluator result JSONs and manifests in datagen's schema (DATASET-v2.md section 6), and stand-in datasets of this
component for the CLI's out and merge-input checks.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
COMP = REPO / "components" / "dataset"
TESTS = COMP / "tests"
for p in (TESTS, REPO / "common", COMP / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
# the image suite's fixtures module is a plugin (pytest_plugins) its test modules also import helpers from
pytest.register_assert_rewrite("real_fixtures")

ROBOT_TASK = {"mobile_panda": "open_fridge_door", "mobile_so101": "open_drawer", "mobile_so101_xl": "open_drawer"}
PREFIXES = ("inputs:", "port:", "frames:", "features:", "task:", "labels:", "episodes:", "merge:")
DROP = object()          # an override value that removes the key


# ------------------------------------------------------------------ the module under test, by path

def load_launch():
    """components/dataset/launch.py as module `dataset_launch` (registered before exec, as ./revolve does)."""
    if "dataset_launch" in sys.modules:
        return sys.modules["dataset_launch"]
    spec = importlib.util.spec_from_file_location("dataset_launch", COMP / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dataset_launch"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("dataset_launch", None)
        raise
    return mod


@pytest.fixture(scope="session")
def launch():
    return load_launch()


# ------------------------------------------------------------------ specs, read independently of the code under test

def contract_path(robot) -> Path:
    return REPO / "specs" / "robots" / robot / "contract.yaml"


def contract_doc(robot) -> dict:
    return yaml.safe_load(contract_path(robot).read_text())


def _channels(node):
    """Every `channel` mapping under node, in file order."""
    if isinstance(node, dict):
        ch = node.get("channel")
        if isinstance(ch, dict) and "topic" in ch:
            yield ch
        for k, v in node.items():
            if k != "channel":
                yield from _channels(v)
    elif isinstance(node, list):
        for v in node:
            yield from _channels(v)


def contract_channels(robot) -> dict:
    """{topic: type} of every channel of the robot's contract, file order, x- anchor holders skipped."""
    out = {}
    for k, v in contract_doc(robot).items():
        if str(k).startswith("x-"):
            continue
        for ch in _channels(v):
            out.setdefault(ch["topic"], ch.get("type", "std_msgs/msg/String"))
    return out


def contract_topics(robot) -> list:
    return list(contract_channels(robot))


def contract_action_topics(robot) -> list:
    """The topics of the contract's `actions` section, file order, no duplicates."""
    out = []
    for ch in _channels(contract_doc(robot).get("actions", {})):
        if ch["topic"] not in out:
            out.append(ch["topic"])
    return out


def prompt(task) -> str:
    return yaml.safe_load((REPO / "specs" / "tasks" / task / "task.yaml").read_text())["prompt"]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ synthetic bags, results, manifests

# Recorded topics that are not in any contract (record.yaml), so a bag looks like the simulator's
EXTRA_TOPICS = {"/clock": "rosgraph_msgs/msg/Clock", "/tf": "tf2_msgs/msg/TFMessage",
                "/tf_static": "tf2_msgs/msg/TFMessage", "/robot_description": "std_msgs/msg/String",
                "/evaluator/result": "std_msgs/msg/String"}


def write_bag_metadata(bag: Path, counts: dict, types: dict | None = None, prompt_text: str | None = None) -> Path:
    """A rosbag2 directory holding only metadata.yaml (rosbag2_bagfile_information.topics_with_message_count)."""
    bag.mkdir(parents=True, exist_ok=True)
    types = types or {}
    info = {
        "version": 9,
        "storage_identifier": "mcap",
        "duration": {"nanoseconds": 20_100_000_000},
        "starting_time": {"nanoseconds_since_epoch": 1_000_000_000},
        "message_count": int(sum(counts.values())),
        "topics_with_message_count": [
            {"topic_metadata": {"name": t, "type": types.get(t, EXTRA_TOPICS.get(t, "std_msgs/msg/String")),
                                "serialization_format": "cdr", "offered_qos_profiles": "",
                                "type_description_hash": "RIHS01_00"},
             "message_count": int(n)}
            for t, n in counts.items()],
        "compression_format": "",
        "compression_mode": "",
        "relative_file_paths": [f"0_{bag.name}.mcap"],
        "files": [],
        "custom_data": {"lerobot.operator_prompt": prompt_text} if prompt_text else {},
        "ros_distro": "lyrical",
    }
    path = bag / "metadata.yaml"
    path.write_text(yaml.safe_dump({"rosbag2_bagfile_information": info}, sort_keys=False))
    return path


def result_doc(task, robot, episode=0, passed=True, duration_s=20.004, extra=None) -> dict:
    d = {"episode": episode, "task": task, "robot": robot, "passed": passed, "duration_s": duration_s,
         "reason": "done" if passed else "approach timeout", "phases": []}
    for k, v in (extra or {}).items():
        if v is DROP:
            d.pop(k, None)
        else:
            d[k] = v
    return d


class Producer:
    """A fake producer run under `root`: per episode `<split>/scene_<seed>/episode_<n>.json` and `episode_<n>_bag/`
    (metadata.yaml only), and the manifest entries listing them (datagen's schema, DATASET-v2.md section 6)."""

    def __init__(self, root: Path, robot="mobile_panda", task="open_fridge_door", topics_of=None):
        self.root, self.robot, self.task = Path(root), robot, task
        self.root.mkdir(parents=True, exist_ok=True)
        self.channels = contract_channels(topics_of or (robot if contract_path(robot).is_file() else "mobile_panda"))
        self.entries: list[dict] = []
        self._numbers: dict = {}

    def episode(self, split="train", seed=0, passed=True, duration_s=20.004, *, drop=(), zero=(), extra_topics=(),
                result=True, result_text=None, result_extra=None, bag=True, block="datagen", attempt=None,
                expert="fake_expert_v1", absolute=False, **overrides) -> dict:
        """One episode on disk + its manifest entry (appended to self.entries and returned).

        drop: contract topics left out of the bag; zero: topics listed with message_count 0; result=False: no result
        file; result_text: the result file's raw text; result_extra: result-JSON overrides (DROP removes a key);
        bag=False: no bag directory, bag="empty": a bag directory without metadata.yaml; block: "datagen" |
        "recovery" | "both" | None; overrides: entry overrides (DROP removes a key)."""
        key = (split, seed)
        n = self._numbers.get(key, 0)
        self._numbers[key] = n + 1
        run = self.root / split / f"scene_{seed}"
        run.mkdir(parents=True, exist_ok=True)
        res, bagdir = run / f"episode_{n}.json", run / f"episode_{n}_bag"
        if result_text is not None:
            res.write_text(result_text)
        elif result:
            res.write_text(json.dumps(result_doc(self.task, self.robot, n, passed, duration_s, extra=result_extra)))
        if bag is True:
            counts = {t: (0 if t in zero else 600) for t in self.channels if t not in drop}
            counts.update({t: 1000 for t in (*EXTRA_TOPICS, *extra_topics)})
            write_bag_metadata(bagdir, counts, {**EXTRA_TOPICS, **self.channels}, prompt(self.task)
                               if (REPO / "specs" / "tasks" / self.task).is_dir() else None)
        elif bag == "empty":
            bagdir.mkdir(parents=True, exist_ok=True)
        rel = (lambda p: str(p)) if absolute else (lambda p: str(p.relative_to(self.root)))
        e = {"seed": seed, "scene": f"{split}/scene_{seed}/scene.yaml", "robot": self.robot, "task": self.task,
             "episode": n, "result": rel(res), "bag": rel(bagdir), "passed": passed}
        att = len(self.entries) if attempt is None else attempt
        if block in ("datagen", "both"):
            e["datagen"] = {"expert": expert, "attempt": att, "wall_s": 41.2}
        if block in ("recovery", "both"):
            e["recovery"] = {"expert": expert, "attempt": att}
        for k, v in overrides.items():
            if v is DROP:
                e.pop(k, None)
            else:
                e[k] = v
        self.entries.append(e)
        return e

    def manifest(self, name="manifest.json", entries=None, **overrides) -> Path:
        entries = self.entries if entries is None else entries
        doc = {"task": self.task, "robot": self.robot, "created": "2026-11-22T10:15:00Z", "episodes": entries,
               "counts": {"scenes": len({e.get("scene") for e in entries}), "attempted": len(entries),
                          "passed": sum(1 for e in entries if e.get("passed") is True)}}
        for k, v in overrides.items():
            if v is DROP:
                doc.pop(k, None)
            else:
                doc[k] = v
        path = self.root / name
        path.write_text(json.dumps(doc, indent=1))
        return path

    def labels(self, entry, manifest, turn=0) -> dict:
        """The labels DATASET-v2.md section 6 derives for a datagen entry (no explicit split field)."""
        return {"source": "demo", "seed": entry["seed"], "split": entry.get("split", entry["scene"].split("/")[0]),
                "scene": entry["scene"], "episode": entry["episode"], "expert": entry["datagen"]["expert"],
                "attempt": entry["datagen"]["attempt"], "turn": turn, "manifest": str(manifest)}


@pytest.fixture
def producer(tmp_path):
    """Factory: producer(robot=..., task=..., root=...) -> Producer under tmp_path."""
    made = []

    def make(robot="mobile_panda", task=None, root=None, **kw):
        p = Producer(root or tmp_path / f"run{len(made)}", robot, task or ROBOT_TASK.get(robot, "open_fridge_door"),
                     **kw)
        made.append(p)
        return p
    return make


def write_manifest(path: Path, prod: Producer, entries) -> Path:
    """A manifest in datagen's schema at any path (entries keep their own, usually absolute, paths)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": prod.task, "robot": prod.robot, "created": "2026-11-22T10:15:00Z",
                                "episodes": entries}, indent=1))
    return path


RECORD = {"kind": "convert", "inputs": {"manifests": ["x"]}, "problems": [], "episodes": 1, "frames": 1}


def fake_dataset(d: Path, **over) -> Path:
    """A finished dataset of this component as far as the CLI's out and merge-input checks see: a
    meta/revolve_dataset.json record (RECORD with overrides, DROP removes a key) and stand-in LeRobot files."""
    (d / "meta").mkdir(parents=True, exist_ok=True)
    doc = dict(RECORD)
    for k, v in over.items():
        if v is DROP:
            doc.pop(k, None)
        else:
            doc[k] = v
    (d / "meta" / "revolve_dataset.json").write_text(json.dumps(doc))
    (d / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30}))
    (d / "data").mkdir(exist_ok=True)
    (d / "data" / "file-000.parquet").write_bytes(b"PAR1 not really")
    return d


def snapshot(path: Path):
    """Everything observable at path: a file's bytes and mtime, or a dir's every entry (dirs, files: bytes, mtime)."""
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_file():
        return ("file", sha256(path), path.stat().st_mtime_ns)
    return ("dir", {str(p.relative_to(path)): ("file", sha256(p), p.stat().st_mtime_ns) if p.is_file() else ("dir",)
                    for p in sorted(path.rglob("*"))})


# ------------------------------------------------------------------ the CLI's output protocol

STAGES = {"checks", "port", "merge", "done", "failed"}


def convert_argv(manifests, out, *extra, turn=0) -> list:
    return ["convert", *[x for m in manifests for x in ("--manifest", str(m))], *extra, "--turn", str(turn),
            "--out", str(out)]


def cli_main(capsys, argv):
    """dataset.cli.main(argv) -> (exit code, stdout); an argparse usage error (SystemExit) gives its code."""
    from dataset import cli
    try:
        rc = cli.main([str(a) for a in argv])
    except SystemExit as e:
        rc = e.code
    return rc, capsys.readouterr().out


def refused(rc, text, prefix) -> list:
    """DATASET-v2 7 / 8.4 on a failure: exit 1, the last stdout line the result JSON {dataset, episodes, frames,
    problems} with a problem of the given prefix, PROGRESS lines of known stages and never `done` -> the problems."""
    assert rc == 1, text
    res = last_json_line(text)
    for k in ("dataset", "episodes", "frames", "problems"):
        assert k in res, (k, res)
    assert res["problems"] and all(isinstance(p, str) for p in res["problems"]), res
    assert any(p.startswith(prefix) for p in res["problems"]), res
    for pr in progress_lines(text):
        assert isinstance(pr, dict) and pr.get("stage") in STAGES and pr.get("stage") != "done", pr
    return res["problems"]


def last_json_line(text: str) -> dict:
    """The output protocol: the LAST stdout line is the result JSON."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines, "no stdout at all"
    try:
        doc = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise AssertionError(f"the last stdout line is not JSON: {lines[-1]!r}") from e
    assert isinstance(doc, dict), f"the last stdout line is not a JSON object: {lines[-1]!r}"
    return doc


def progress_lines(text: str) -> list:
    return [json.loads(ln[len("PROGRESS "):]) for ln in text.splitlines() if ln.startswith("PROGRESS ")]


def tree_mtimes(root) -> dict:
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in Path(root).rglob("*") if p.is_file()}


def no_env_image():
    return "REVOLVE_DATASET_IMAGE" not in os.environ
