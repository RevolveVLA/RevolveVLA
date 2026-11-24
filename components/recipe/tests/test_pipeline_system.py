"""The recipe end to end on the fake runs: the CLI's stages, selection.json, report.md, exit codes."""
import json
from pathlib import Path

import pytest
import yaml

from recipe import cli

SOURCES = {"base": {"plugin": "datagen", "manifest": "datagen/manifest.json"},
           "rollouts": {"plugin": "rollouts", "run": "eval", "passed": True},
           "recovery": {"plugin": "recovery", "run": "recovery", "kind": "phase"},
           "dagger": {"plugin": "recovery", "run": "recovery", "kind": "dagger", "segment": "interventions"}}
RECIPE = {"seed": 0, "sources": SOURCES,
          "take": {"base": "all", "rollouts": 2, "recovery": {"approach": 2, "grasp": 1}, "dagger": {"grasp": 1, "open": 1}},
          "splits": {"train": 0.5, "val": 0.5}, "exclude": ["eval/heldout"], "out": "out"}


def run(root, capsys, stage="materialise", extra=(), **edits):
    recipe = {**RECIPE, **edits}
    path = root / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    rc = cli.main([str(path), stage, *extra], root=root)
    out = capsys.readouterr()
    return rc, out.out, out.err


def selection(root):
    return json.loads((root / "out" / "selection.json").read_text())


def items(sel):
    return [i for part in sel["splits"].values() for i in part]


def test_materialise_writes_the_selection_and_the_report(root, capsys):
    rc, out, _ = run(root, capsys)
    assert rc == 0 and "wrote" in out
    sel = selection(root)
    assert sel["recipe"]["unit"] == "episodes" and sel["recipe"]["splits"] == {"train": 0.5, "val": 0.5}
    assert sel["refused"] == {"base": 0, "rollouts": 1, "recovery": 1, "dagger": 0}
    got = {(c["source"], c["phase"]): (c["asked"], c["got"], sum(c["splits"].values())) for c in sel["counts"]}
    assert got == {("base", None): ("all", 3, 3), ("rollouts", None): (2, 2, 2), ("recovery", "approach"): (2, 2, 2),
                   ("recovery", "grasp"): (1, 1, 1), ("dagger", "grasp"): (1, 1, 1), ("dagger", "open"): (1, 1, 1)}
    report = (root / "out" / "report.md").read_text()
    assert "| recovery | phase | approach | 2 | 2 | 2 |" in report and "Refused by the leakage guard" in report


def test_item_fields(root, capsys):
    run(root, capsys)
    sel = selection(root)
    dagger = [i for i in items(sel) if i["kind"] == "dagger"]
    assert sorted((i["phase"], i["segment"], i["frames"]) for i in dagger) == [("grasp", [8.0, 30.0], 660), ("open", [12.0, 40.0], 840)]
    for i in items(sel):
        assert set(i) >= {"kind", "source", "phase", "scene", "config_seed", "episode_dir", "result", "bag", "segment", "passed",
                          "labels"} and Path(i["bag"]).is_dir() and Path(i["scene"]).is_absolute()


def test_no_excluded_scene_reaches_the_selection(root, capsys):
    run(root, capsys, take={**RECIPE["take"], "rollouts": "all", "recovery": {"approach": 99, "grasp": 99}})
    assert all("heldout" not in i["scene"] and i["config_seed"] not in (500, 501) for i in items(selection(root)))
    run(root, capsys, exclude=[])
    assert {i["config_seed"] for i in items(selection(root))} >= {500}


def test_one_scene_is_in_one_split(root, capsys):
    run(root, capsys)
    owner = {}
    for name, part in selection(root)["splits"].items():
        for i in part:
            assert owner.setdefault(i["scene"], name) == name


def test_deterministic_and_seeded(root, capsys):
    run(root, capsys, take={**RECIPE["take"], "rollouts": 1})
    a = selection(root)
    run(root, capsys, take={**RECIPE["take"], "rollouts": 1})
    b = selection(root)
    assert (a["splits"], a["counts"]) == (b["splits"], b["counts"])
    picks = set()
    for seed in range(8):
        run(root, capsys, seed=seed, take={**RECIPE["take"], "rollouts": 1})
        picks.add(tuple(i["result"] for i in selection(root)["splits"]["train"] if i["source"] == "rollouts"))
    assert len(picks) > 1


def test_frames_unit(root, capsys):
    run(root, capsys, unit="frames", fps=10, take={"base": 450, "rollouts": "all", "recovery": 0, "dagger": {"open": 100}})
    sel = selection(root)
    by = {(c["source"], c["phase"]): c for c in sel["counts"]}
    assert by[("base", None)]["got"] >= 450 and by[("base", None)]["got"] % 10 == 0 and by[("dagger", "open")]["got"] == 280
    assert sel["recipe"]["fps"] == 10 and {i["frames"] for i in items(sel) if i["kind"] == "dagger"} == {280}


def test_short_warns_and_writes(root, capsys):
    rc, out, _ = run(root, capsys, take={**RECIPE["take"], "dagger": {"approach": 3, "open": 1}})
    assert rc == 0 and "WARN" in out and "SHORT" in out and (root / "out" / "selection.json").is_file()
    assert any(c["phase"] == "approach" and c["got"] == 0 for c in selection(root)["counts"] if c["source"] == "dagger")


def test_short_with_on_short_error_fails_but_reports(root, capsys):
    rc, out, _ = run(root, capsys, on_short="error", take={**RECIPE["take"], "dagger": {"approach": 3}})
    assert rc == 1 and "FAIL" in out and "Short: dagger/approach" in (root / "out" / "report.md").read_text()
    assert not (root / "out" / "selection.json").exists()


def test_check_and_resolve_write_nothing(root, capsys):
    rc, out, _ = run(root, capsys, "check")
    assert rc == 0 and "| source | kind | phase |" in out and not (root / "out").exists()
    rc, out, _ = run(root, capsys, "resolve")
    assert rc == 0 and "base (datagen): 3 usable" in out and "dagger (dagger): 2 usable: grasp 1, open 1" in out
    assert "refused 1" in out and not (root / "out").exists()


def test_dry_run_reads_no_run(root, capsys):
    rc, out, _ = run(root, capsys, extra=["--dry-run"], sources={**SOURCES, "base": {"plugin": "datagen", "manifest": "not/there.json"}})
    assert rc == 0 and "recipe " in out and "not/there.json" in out and not (root / "out").exists()


def test_errors_and_exit_codes(root, capsys):
    rc, _, err = run(root, capsys, unit="steps")
    assert rc == 2 and "unit" in err
    rc, _, err = run(root, capsys, sources={**SOURCES, "rollouts": {"plugin": "rollouts", "run": "missing"}})
    assert rc == 1 and "missing" in err
    rc, _, err = run(root, capsys, exclude=["eval/missing"])
    assert rc == 1 and "no scene.yaml" in err
    rc, _, err = run(root, capsys, sources={**SOURCES, "base": {"plugin": "nope"}})
    assert rc == 1 and "nope" in err


def test_an_external_plugin_is_found_on_the_plugin_path(root, capsys, monkeypatch, tmp_path):
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "oneof.py").write_text(
        "from recipe.core.interfaces import Source\nfrom recipe.core.items import Read\nfrom recipe.core.registry import REGISTRY\n"
        "@REGISTRY.register('source', 'oneof')\nclass S(Source):\n    kind = 'x'\n    def __init__(self, name, params, root): pass\n"
        "    def read(self): return Read([])\n")
    monkeypatch.setenv("RECIPE_PLUGIN_PATH", str(plug))
    rc, out, _ = run(root, capsys, "resolve", sources={"x": {"plugin": "oneof"}}, take={"x": "all"})
    assert rc == 0 and "x (x): 0 usable" in out
