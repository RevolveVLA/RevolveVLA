"""Fail closed on real inputs in the dataset image (DATASET-v2.md sections 3, 5.4, 7, 8.3, 8.4, 9): a real bag missing
a contract topic, a bag rosbag2 cannot read, a gripper stream Rosetta cannot decode (with its untouched control),
--out on an input (the producer's run dir, a merge input), and a merge input whose record says its build failed (with
its control). Each refusal exits 1 with the result line's problems and leaves nothing at <out>. Session fixtures:
real_fixtures.py."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from conftest import convert_argv, last_json_line
from real_fixtures import (HAND, copies, failed_closed, first_entry, light_snapshot, link_bag, rewritten_run, run_cli,
                           tail, write_run_manifest)

pytest_plugins = ["real_fixtures"]


# ------------------------------------------------------------------ bags

def test_real_bag_missing_joint_trajectory(ready, fixtures, outdir):
    # DATASET-v2 5.4/8.3: a passed entry whose real bag lacks /panda_arm_controller/joint_trajectory fails the whole
    # conversion with an inputs: problem naming the topic and the manifest entry; nothing built
    m, (e,) = first_entry(fixtures)
    src = m.parent / e["bag"]
    root = outdir / "missing_topic"
    man = write_run_manifest(root, m, [e])
    bag = root / "episode_0_bag"
    bag.mkdir()
    for f in src.iterdir():
        if f.name != "metadata.yaml":
            (bag / f.name).symlink_to(f.resolve())
    meta = yaml.safe_load((src / "metadata.yaml").read_text())
    info = meta["rosbag2_bagfile_information"]
    topic = "/panda_arm_controller/joint_trajectory"
    before = len(info["topics_with_message_count"])
    info["topics_with_message_count"] = [t for t in info["topics_with_message_count"]
                                         if t["topic_metadata"]["name"] != topic]
    assert len(info["topics_with_message_count"]) == before - 1, "the real bag should have had the topic"
    (bag / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    out = root / "ds"
    rc, text, _, _ = run_cli(ready, convert_argv([man], out), outdir / "logs" / "missing_topic")
    probs = failed_closed(rc, text, out, "inputs:")
    assert any(p.startswith("inputs:") and topic in p and f"{man} entry 0" in p for p in probs), probs
    assert not out.with_name(out.name + ".partial").exists(), "an input failure builds nothing"


@pytest.mark.parametrize("mode", ["missing", "truncated"])
def test_unreadable_bag_is_an_input_problem(ready, fixtures, outdir, mode):
    # entry 1's bag keeps its real metadata.yaml (the topic check passes) but its .mcap is missing or truncated to 100
    # bytes: every bag is read before the port, so this is an inputs: problem naming that entry, nothing is built, the
    # last stdout line is still the result JSON and no traceback ends the run
    m, es = first_entry(fixtures, n=2)
    root = outdir / f"unreadable_{mode}"
    man = write_run_manifest(root, m, es)
    link_bag(m.parent / es[0]["bag"], root / "episode_0_bag")
    src, dst = m.parent / es[1]["bag"], root / "episode_1_bag"
    dst.mkdir()
    shutil.copy(src / "metadata.yaml", dst / "metadata.yaml")
    if mode == "truncated":
        for f in src.glob("*.mcap"):
            with open(f.resolve(), "rb") as fh:
                (dst / f.name).write_bytes(fh.read(100))
    out = root / "ds"
    rc, text, _, err = run_cli(ready, convert_argv([man], out), outdir / "logs" / f"unreadable_{mode}")
    probs = failed_closed(rc, text, out, "inputs:")
    assert any(p.startswith("inputs:") and f"{man} entry 1" in p for p in probs), probs
    assert not out.with_name(out.name + ".partial").exists(), "an input failure builds nothing"
    err_lines = [ln for ln in Path(err).read_text(errors="replace").splitlines() if ln.strip()]
    # an uncaught exception ends stderr with "<Type>Error: ..." after a traceback
    assert not (err_lines and re.match(r"^[A-Za-z_][\w.]*(Error|Exception)\b", err_lines[-1])
                and any(ln.startswith("Traceback (most recent call last)") for ln in err_lines)), \
        "a traceback ends the run:\n" + "\n".join(err_lines[-20:])


def test_undecodable_hand_stream_fails(ready, fixtures, outdir):
    # every hand joint_trajectory message names a joint the contract does not select: Rosetta logs "Decode failed for
    # 'action' on /panda_hand_controller/joint_trajectory (...); dropping message" once and zero-fills the gripper
    # action, so convert exits 1 with a port: problem quoting it and nothing at out
    root = outdir / "hand_tampered"
    man, r = rewritten_run(ready, fixtures, root, "tamper")
    assert r["changed"] > 0, r
    out = root / "ds"
    rc, text, _, _ = run_cli(ready, convert_argv([man], out), outdir / "logs" / "hand_tampered")
    probs = failed_closed(rc, text, out, "port:")
    assert any(p.startswith("port:") and "Decode failed" in p and HAND in p for p in probs), probs


def test_rewritten_untouched_bag_converts(ready, fixtures, outdir):
    # control of the case above: the same rosbag2 rewrite without the tamper converts cleanly, so that failure is the
    # decode failure, not the rewrite (and benign Rosetta lines are not problems)
    root = outdir / "hand_plain"
    man, r = rewritten_run(ready, fixtures, root, "plain")
    assert r["changed"] == 0 and r["dropped"] == 0, r
    out = root / "ds"
    rc, text, _, _ = run_cli(ready, convert_argv([man], out), outdir / "logs" / "hand_plain")
    assert rc == 0, tail(text)
    assert last_json_line(text)["problems"] == []


# ------------------------------------------------------------------ --out on an input

def test_out_is_the_run_dir(ready, fixtures, outdir):
    # `convert --manifest <run>/manifest.json --out <run>`: an out: problem, every file of the run still there (sizes,
    # mtimes), no <run>.partial
    m, (e,) = first_entry(fixtures)
    run = outdir / "out_is_run"
    man = write_run_manifest(run, m, [e])
    shutil.copytree(m.parent / e["bag"], run / "episode_0_bag")
    before = light_snapshot(run)
    rc, text, _, _ = run_cli(ready, convert_argv([man], run), outdir / "logs" / "out_is_run")
    assert rc == 1, tail(text)
    assert any(p.startswith("out:") for p in last_json_line(text)["problems"]), tail(text)
    assert light_snapshot(run) == before, "the run dir was modified"
    assert not run.with_name(run.name + ".partial").exists()


def test_merge_out_is_an_input(ready, converted, outdir):
    # `merge --out A A B`: an out: problem, A and B unchanged (copies of real datasets, never the shared ones)
    a, b = copies(converted, outdir / "merge_out_is_input", "train", "val")
    before, bbefore = light_snapshot(a), light_snapshot(b)
    rc, text, _, _ = run_cli(ready, ["merge", "--out", str(a), str(a), str(b)], outdir / "logs" / "merge_out_is_input")
    assert rc == 1, tail(text)
    assert any(p.startswith("out:") for p in last_json_line(text)["problems"]), tail(text)
    assert light_snapshot(a) == before and light_snapshot(b) == bbefore
    assert not a.with_name(a.name + ".partial").exists()


# ------------------------------------------------------------------ merge inputs must be successful builds

def test_merge_refuses_a_failed_build(ready, converted, outdir):
    # two real datasets, one whose meta/revolve_dataset.json says its build failed (non-empty problems): only the
    # record can refuse it, so merge exits 1 with a merge: problem naming that input and nothing at out
    root = outdir / "merge_failed_build"
    a, b = copies(converted, root, "train", "val")
    f = b / "meta" / "revolve_dataset.json"
    f.write_text(json.dumps({**json.loads(f.read_text()),
                             "problems": ["frames: m.json entry 0: 640 frames, the bag's ticks minus warmup are 600"]}))
    out = root / "merged"
    rc, text, _, _ = run_cli(ready, ["merge", "--out", str(out), str(a), str(b)], outdir / "logs" / "merge_failed")
    probs = failed_closed(rc, text, out, "merge:")
    assert any(p.startswith("merge:") and str(b) in p for p in probs), probs


def test_merge_of_untouched_copies_succeeds(ready, converted, outdir):
    # control of the case above: the same copies with their records untouched merge cleanly, so that refusal is the
    # record check, not the copying
    root = outdir / "merge_copies"
    a, b = copies(converted, root, "train", "val")
    out = root / "merged"
    rc, text, _, _ = run_cli(ready, ["merge", "--out", str(out), str(a), str(b)], outdir / "logs" / "merge_copies")
    assert rc == 0, tail(text)
    assert last_json_line(text)["problems"] == []
    assert (out / "meta" / "revolve_dataset.json").is_file()
