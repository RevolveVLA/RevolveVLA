"""core/spec.py: parsing, defaults, errors naming the yaml path, the hash, the repo's example recipe."""
import copy
from pathlib import Path

import pytest

from datagen.core.spec import SpecError
from recipe.core import spec as S

REPO = Path(__file__).resolve().parents[3]
RAW = {"sources": {"base": {"plugin": "datagen", "manifest": "m.json"}, "rec": {"plugin": "recovery", "run": "r", "kind": "phase"}},
       "take": {"base": "all", "rec": {"approach": 3, "grasp": "all"}}, "out": "runs/recipes/x"}


def parse(**edits):
    raw = copy.deepcopy(RAW)
    for path, value in edits.items():
        d = raw
        *keys, last = path.split("__")
        for k in keys:
            d = d[k]
        d[last] = value
    return S.parse(raw, "r.yaml")


def test_defaults_and_take_forms():
    s = parse()
    assert (s.seed, s.unit, s.fps, s.on_short, s.exclude) == (0, "episodes", 30, "warn", ())
    assert dict(s.splits) == {"train": 1.0} and s.output.plugin == "selection"
    assert s.sources["rec"].params == {"kind": "phase", "run": "r"}
    assert s.take == {"base": {None: "all"}, "rec": {"approach": 3, "grasp": "all"}}
    assert parse(take__base=7).take["base"] == {None: 7}


def test_the_repo_example_loads():
    s = S.load(REPO / "specs" / "recipes" / "fridge_round1.yaml")
    assert set(s.sources) == set(s.take) == {"base", "rollouts", "recovery", "dagger"}
    assert dict(s.splits) == {"train": 0.9, "val": 0.1} and s.exclude == ("runs/eval/fridge_r0/heldout",)


@pytest.mark.parametrize("edits, at", [
    ({"take__ghost": 1}, "take.ghost"), ({"take__base": -1}, "take.base"), ({"take__base": True}, "take.base"),
    ({"take__rec": {}}, "take.rec"), ({"take__rec": {"approach": "some"}}, "take.rec.approach"), ({"unit": "steps"}, "unit"),
    ({"on_short": "ignore"}, "on_short"), ({"fps": 0}, "fps"), ({"splits": {"train": 0.5}}, "splits"),
    ({"splits": {"train": 1.5, "val": -0.5}}, "splits.val"), ({"splits": {}}, "splits"), ({"exclude": "runs/x"}, "exclude"),
    ({"sources__base": {"manifest": "m"}}, "sources.base"), ({"colour": 1}, "colour"), ({"sources": {}}, "sources")])
def test_errors_name_the_yaml_path(edits, at):
    with pytest.raises(SpecError, match=at.replace(".", r"\.")):
        parse(**edits)


def test_a_source_without_an_amount_is_refused():
    raw = copy.deepcopy(RAW)
    del raw["take"]["rec"]
    with pytest.raises(SpecError, match="no amount for source 'rec'"):
        S.parse(raw)


def test_hash_ignores_out_only():
    assert parse().hash == parse(out="elsewhere").hash
    assert parse().hash != parse(seed=1).hash and parse().hash != parse(take__base=2).hash


def test_unreadable_yaml(tmp_path):
    with pytest.raises(SpecError, match="cannot read"):
        S.load(tmp_path / "missing.yaml")
