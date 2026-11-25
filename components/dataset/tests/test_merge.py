"""Real merges in the dataset image (DATASET-v2.md section 9): mobile_panda's train and val conversions merged in that
order, loaded back with their labels, refused when the inputs' contracts or robots differ, and built from the inputs'
own contract rather than the checkout's. Session fixtures: real_fixtures.py."""
from __future__ import annotations

import json

from conftest import contract_path, last_json_line, prompt
from real_fixtures import FPS, copies, failed_closed, load_in_image, run_cli, tail

pytest_plugins = ["real_fixtures"]

MERGE_CHECKS = {"merge", "frames", "features", "task", "labels", "episodes"}


def test_merge(merged):
    # merge in the given order: labels and lengths concatenated, frames summed, kind merge, contract copied from the
    # first input, exactly the merge checks, all ok, nothing left in .partial
    c, t, v = merged
    assert c.rc == 0, c.why()
    res = last_json_line(c.text)
    st, sv, s = t.summary(), v.summary(), c.summary()
    assert res["problems"] == [] and res["episodes"] == 3 and res["dataset"] == str(c.out)
    assert res["frames"] == st["frames"] + sv["frames"] == s["frames"]
    assert s["kind"] == "merge"
    assert str(t.out) in json.dumps(s["inputs"]) and str(v.out) in json.dumps(s["inputs"])
    assert [p["labels"] for p in s["per_episode"]] == [p["labels"] for p in st["per_episode"] + sv["per_episode"]]
    assert [p["frames"] for p in s["per_episode"]] == [p["frames"] for p in st["per_episode"] + sv["per_episode"]]
    assert set(s["checks"]) == MERGE_CHECKS and all(x == "ok" for x in s["checks"].values()), s["checks"]
    info = c.info()
    assert info["total_episodes"] == 3 and info["total_frames"] == s["frames"] and info["fps"] == FPS
    assert (c.out / "meta" / "rosetta_contract.yaml").read_bytes() == (t.out / "meta" / "rosetta_contract.yaml"
                                                                       ).read_bytes()
    assert not c.out.with_name(c.out.name + ".partial").exists()


def test_merge_loads_with_labels(ready, merged):
    # labels preserved through aggregate_datasets (read back from meta/episodes), lengths likewise
    c, t, v = merged
    assert c.rc == 0, c.why()
    r = load_in_image(ready, c.out)
    st, sv = t.summary(), v.summary()
    assert r["labels"] == [p["labels"] for p in st["per_episode"] + sv["per_episode"]]
    assert r["lengths"] == [p["frames"] for p in st["per_episode"] + sv["per_episode"]]
    assert r["len"] == st["frames"] + sv["frames"] and r["tasks"] == {"0": prompt("open_fridge_door")}


def test_merge_contract_differs(ready, converted, outdir):
    # fail closed unless every input's meta/rosetta_contract.yaml is byte-identical
    t = converted("mobile_panda", "train")
    assert t.rc == 0, t.why()
    (altered,) = copies(converted, outdir / "contract_changed", "val")
    f = altered / "meta" / "rosetta_contract.yaml"
    data = bytearray(f.read_bytes())
    data[2] = ord("M") if data[2] != ord("M") else ord("m")            # one byte, in the first comment line
    f.write_bytes(bytes(data))
    out = outdir / "merge_contract_differs"
    rc, text, _, _ = run_cli(ready, ["merge", "--out", str(out), str(t.out), str(altered)],
                             outdir / "logs" / "merge_contract_differs")
    failed_closed(rc, text, out, "merge:")


def test_merge_robots_differ(ready, converted, outdir):
    # fail closed on different robots (contracts, features)
    a, b = converted("mobile_panda", "val"), converted("mobile_so101")
    assert a.rc == 0 and b.rc == 0, (a.why(), b.why())
    out = outdir / "merge_robots_differ"
    rc, text, _, _ = run_cli(ready, ["merge", "--out", str(out), str(a.out), str(b.out)],
                             outdir / "logs" / "merge_robots_differ")
    failed_closed(rc, text, out, "merge:")


def test_merge_uses_the_inputs_contract(ready, converted, outdir):
    # two datasets whose meta/rosetta_contract.yaml are identical to each other but differ from the checkout's
    # specs/robots/mobile_panda/contract.yaml (a comment appended to both): merge exits 0 and the merged
    # meta/rosetta_contract.yaml is the inputs' file, so a contract edited after a conversion does not break its merge
    root = outdir / "own_contract"
    a, b = copies(converted, root, "train", "val")
    for d in (a, b):
        f = d / "meta" / "rosetta_contract.yaml"
        f.write_bytes(f.read_bytes() + b"# a comment the checkout's contract does not have\n")
    want = (a / "meta" / "rosetta_contract.yaml").read_bytes()
    assert want == (b / "meta" / "rosetta_contract.yaml").read_bytes() != contract_path("mobile_panda").read_bytes()
    out = root / "merged"
    rc, text, _, _ = run_cli(ready, ["merge", "--out", str(out), str(a), str(b)], outdir / "logs" / "own_contract")
    assert rc == 0, tail(text)
    assert last_json_line(text)["problems"] == []
    assert (out / "meta" / "rosetta_contract.yaml").read_bytes() == want
