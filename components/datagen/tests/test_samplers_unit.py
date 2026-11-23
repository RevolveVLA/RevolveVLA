"""core/samplers.py: determinism, ranges, overrides, stratified balance, sobol / lhs spread, redraw, min_distance, summary."""
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from datagen.core import registry, samplers as S
from datagen.core.interfaces import Config
from datagen.core.spec import Choices, MinDistance, PoolSpec, Range, SamplerSpec, SplitSpec

X, Y, YAW, G = "r.x", "r.y", "r.yaw", "r.grasp"
DIMS = {X: Range(-0.17, 0.17), Y: Range(-0.1, 0.1), YAW: Range(-27, 27), G: Choices((0, 1, 2, 3, 4))}


def split(strategy="random", name="train", target=10, bins=None, overrides=None, md=None):
    return SplitSpec(name, target, SamplerSpec(strategy, bins or {}, overrides or {}), PoolSpec((0, 99)), md)


def run(strategy="random", seeds=range(30), base=0, dims=DIMS, **kw):
    reject, avoid = kw.pop("reject", None), kw.pop("avoid", ())
    s = registry.create("sampler", strategy)
    return s, s.sample(split(strategy, **kw), dims, list(seeds), base, reject, avoid)


def spread(values, options):
    c = Counter(values)
    return max(c.get(o, 0) for o in options) - min(c.get(o, 0) for o in options)


@pytest.mark.parametrize("strategy", ["random", "stratified", "sobol", "lhs"])
def test_deterministic_in_ranges_and_json_values(strategy):
    a, b = run(strategy)[1], run(strategy)[1]
    assert a == b and a != run(strategy, base=1)[1]
    assert [c.seed for c in a] == list(range(30)) and all(c.split == "train" for c in a)
    for c in a:
        assert set(c.values) == set(DIMS)
        assert all(DIMS[k].lo <= v <= DIMS[k].hi for k, v in c.values.items() if k != G) and c.values[G] in range(5)


def test_random_seed_independent_of_pool():
    a = {c.seed: c.values for c in run("random", range(10))[1]}
    b = {c.seed: c.values for c in run("random", range(5, 40))[1]}
    assert all(a[s] == b[s] for s in range(5, 10))


def test_overrides_fix_dims_and_are_never_drawn():
    _, cs = run(overrides={YAW: 5.0, G: 3})
    assert {c.values[YAW] for c in cs} == {5.0} and {c.values[G] for c in cs} == {3}
    plain = run()[1]
    assert len({c.values[X] for c in cs}) == 30 and len({c.values[X] for c in plain}) == 30


@pytest.mark.parametrize("n", [30, 37, 7])
def test_stratified_balance_and_cells(n):
    _, cs = run("stratified", range(100, 100 + n), bins={YAW: 3})
    assert all(len(c.cell) == 4 for c in cs)              # one index per varied dim, in dims order
    assert max(Counter(c.cell for c in cs).values()) == -(-n // 15)
    assert spread([c.cell[3] for c in cs], range(5)) <= 1 and spread([c.cell[2] for c in cs], range(3)) <= 1
    for c in cs:
        assert c.cell[0] == c.cell[1] == 0                # unbinned dims: one cell
        lo = -27 + 18 * c.cell[2]
        assert lo <= c.values[YAW] <= lo + 18 and c.values[G] == c.cell[3]


def test_stratified_full_cycle_covers_every_cell_once():
    _, cs = run("stratified", range(15), bins={YAW: 3})
    assert sorted(Counter(c.cell for c in cs).values()) == [1] * 15


def test_stratified_prefix_is_balanced():
    _, cs = run("stratified", range(30), bins={YAW: 3})
    first = cs[:10]
    assert sorted(Counter(c.cell[3] for c in first).values()) == [2] * 5
    assert sorted(Counter(c.cell[2] for c in first).values()) == [3, 3, 4]


def test_stratified_overridden_dim_has_no_cell():
    _, cs = run("stratified", range(10), bins={YAW: 2}, overrides={G: 1})
    assert all(len(c.cell) == 3 for c in cs) and {c.values[G] for c in cs} == {1}


def test_sobol_and_lhs_are_better_spread_than_random():
    dims = {X: Range(0, 1), Y: Range(0, 1)}
    spreads = {s: S.summary(run(s, range(64), dims=dims)[1], dims)["min_distance"] for s in ("random", "sobol", "lhs")}
    assert spreads["sobol"] > spreads["random"] and spreads["lhs"] > spreads["random"] * 0.5
    sob = run("sobol", range(64), dims=dims)[1]
    for d in (X, Y):                                          # a scrambled Sobol net: 8 per eighth
        assert sorted(Counter(int(c.values[d] * 8) for c in sob).values()) == [8] * 8


def test_lhs_picks_are_a_hypercube_and_replacements_another():
    dims = {X: Range(0, 1), Y: Range(0, 1)}
    _, cs = run("lhs", range(30), dims=dims, target=10)
    for part in (cs[:10], cs[10:]):
        for d in (X, Y):
            assert sorted(int(c.values[d] * len(part)) for c in part) == list(range(len(part)))


def test_sobol_prefix_balanced_over_choices():
    dims = {G: Choices((0, 1, 2, 3))}
    _, cs = run("sobol", range(32), dims=dims)
    assert sorted(Counter(c.values[G] for c in cs[:16]).values()) == [4] * 4


@pytest.mark.parametrize("strategy", ["random", "stratified", "sobol", "lhs"])
def test_reject_redraws_in_cell_and_records(strategy):
    reject = lambda c: "collision" if c.values[X] > 0 else None
    s, cs = run(strategy, bins={YAW: 3}, reject=reject)
    assert all(c.values[X] <= 0 for c in cs)
    assert s.redrawn and all(r["reason"] == "collision" and r["values"][X] > 0 for r in s.redrawn)
    cell = {c.seed: list(c.cell) for c in cs}
    assert all(r["cell"] == cell[r["seed"]] for r in s.redrawn)
    plain = {c.seed: c for c in run(strategy, bins={YAW: 3})[1]}
    assert all(c == plain[c.seed] for c in cs if c.seed not in {r["seed"] for r in s.redrawn})
    assert run(strategy, bins={YAW: 3})[0].redrawn == []


def test_reject_exhausted_names_seed_cell_and_reason():
    with pytest.raises(S.SamplingError, match=r"seed 3 \(cell \[\]\): all 100 draws were rejected, last: no IK"):
        run(seeds=[3], reject=lambda c: "no IK")


def test_min_distance_to_another_split():
    avoid = [Config("train", 100 + i, (), {X: x, Y: 0.0, YAW: 0.0, G: 0}) for i, x in enumerate((-0.1, 0.0, 0.1))]
    md = MinDistance("train", 0.6)
    s, cs = run(seeds=range(10), md=md, avoid=avoid, name="heldout")
    assert all(S.distance(c.values, a.values, DIMS) >= 0.6 for c in cs for a in avoid)
    assert any(r["reason"] == "min_distance" for r in s.redrawn)
    with pytest.raises(S.SamplingError, match="last: min_distance"):
        run(seeds=range(3), md=MinDistance("train", 5.0), avoid=avoid, name="heldout")


def test_min_distance_within_own_split():
    s, cs = run("sobol", range(20), md=MinDistance("train", 0.3))
    pair = S.summary(cs, DIMS)["min_distance"]
    assert pair >= 0.3 and len(cs) == 20


def test_avoid_is_ignored_without_min_distance():
    avoid = [Config("train", 0, (), dict(run()[1][0].values))]
    assert run(avoid=avoid)[1] == run()[1]


def test_distance_is_normalised_ranges_and_zero_or_one_per_choices_dim():
    a = {X: 0.17, Y: 0.0, YAW: 0.0, G: 0}
    b = {X: -0.17, Y: 0.1, YAW: 0.0, G: 4}
    assert S.distance(a, b, DIMS) == pytest.approx((1 + 0.25 + 1) ** 0.5)            # a different choice is one full range
    assert S.distance(a, b | {G: 0}, DIMS) == pytest.approx((1 + 0.25) ** 0.5)       # the same choice adds nothing
    assert S.distance(a, a, {**DIMS, "z": Range(1, 1), "c": Choices((7,))}) == 0
    assert S.distance({G: 1}, {G: 3}, {G: DIMS[G]}) == 1 == S.distance({G: 0}, {G: 4}, {G: DIMS[G]})   # any two differ alike
    assert S.distance({G: 1}, {X: 0.0}, DIMS) == 0                                       # only the dims both have


def test_a_choices_only_split_keeps_its_min_distance():
    """grasp only (no continuous dim): it was `always too close`; now a different choice is distance 1."""
    dims = {G: DIMS[G]}
    avoid = [Config("train", 100, (), {G: 0}), Config("train", 101, (), {G: 1})]
    s, cs = run(seeds=range(8), dims=dims, md=MinDistance("train", 0.5), avoid=avoid, name="heldout")
    assert {c.values[G] for c in cs} <= {2, 3, 4} and any(r["reason"] == "min_distance" for r in s.redrawn)
    with pytest.raises(S.SamplingError, match="last: min_distance"):
        run(seeds=range(3), dims=dims, md=MinDistance("train", 1.5), avoid=avoid, name="heldout")     # 1 < 1.5: none is far enough
    _, own = run(seeds=range(4), dims=dims, md=MinDistance("train", 0.5))                # within one split: all different
    assert len({c.values[G] for c in own}) == 4


def test_summary_counts_and_distances():
    _, cs = run("stratified", range(30), bins={YAW: 3})
    out = S.summary(cs, DIMS, {YAW: 3}, others=cs[:1])
    assert out["count"] == 30 and sum(out["cells"].values()) == 30 and set(out["cells"].values()) == {2}
    assert out["dims"][YAW]["bin_counts"] == [10, 10, 10] and out["dims"][G]["counts"] == {g: 6 for g in range(5)}
    assert out["min_distance"] > 0 and out["min_distance_to"] == pytest.approx(
        min(S.distance(c.values, cs[0].values, DIMS) for c in cs[1:]))
    assert S.summary([], DIMS)["min_distance"] is None


def test_lazy_scipy_random_and_stratified_import_no_numpy():
    code = ("import sys; sys.path[:0] = [sys.argv[1]]\n"
            "from datagen.core import registry\n"
            "from datagen.core.spec import *\n"
            "d = {'a': Range(0, 1), 'g': Choices((1, 2))}\n"
            "for st in ('random', 'stratified'):\n"
            "    sp = SplitSpec('t', 2, SamplerSpec(st, {'a': 2}), PoolSpec((0, 9)))\n"
            "    assert len(registry.create('sampler', st).sample(sp, d, range(6), 0)) == 6\n"
            "bad = [m for m in ('numpy', 'scipy') if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve().parents[1] / "src")], check=True)
