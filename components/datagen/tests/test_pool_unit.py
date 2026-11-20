"""core/pool.py walk: picks, in-cell replacement, reasons, exhaustion, resume (pure, no plugins)."""
import pytest

from datagen.core.interfaces import Config, Result
from datagen.core.pool import PoolExhausted, walk


def cfg(seed, cell=()):
    return Config("s", seed, cell, {})


def rec():
    return {"used": [], "rejected": []}


def run(configs, target, fail=(), r=None, calls=None):
    r = rec() if r is None else r

    def attempt(c):
        if calls is not None:
            calls.append(c.seed)
        return ("generate", Result.fail(f"bad {c.seed}")) if c.seed in fail else ("generate", Result())
    walk(configs, r, target, attempt, lambda: None)
    return r


def used(r):
    return [e["config"]["seed"] for e in r["used"]]


def two_cells(n=12):
    return [cfg(i, (i % 2,)) for i in range(n)]       # even seeds cell 0, odd seeds cell 1


def test_no_failure_takes_the_first_target_configs():
    assert used(run(two_cells(), 4)) == [0, 1, 2, 3]


def test_cell_less_replacement_is_the_next_unused_config_in_order():
    r = run([cfg(i) for i in range(8)], 3, fail={0, 1})
    assert used(r) == [2, 3, 4] and r["rejected"] == [{"seed": 0, "step": "generate", "reason": "bad 0"},
                                                      {"seed": 1, "step": "generate", "reason": "bad 1"}]


def test_replacement_stays_in_the_cell_of_the_rejected_pick():
    r = run(two_cells(), 4, fail={1})                  # picks 0 1 2 3; 1 (cell 1) -> next unused cell-1 seed: 5
    assert sorted(used(r)) == [0, 2, 3, 5]
    r = run(two_cells(), 4, fail={1, 5})
    assert sorted(used(r)) == [0, 2, 3, 7]


def test_replacement_chain_and_reasons_per_step():
    def attempt(c):
        return ("validate", Result.fail("wall")) if c.seed == 0 else (("generate", Result.fail("no IK")) if c.seed == 2
                                                                         else ("generate", Result()))
    r = rec()
    walk(two_cells(), r, 3, attempt, lambda: None)
    assert [(e["seed"], e["step"], e["reason"]) for e in r["rejected"]] == [(0, "validate", "wall"),
                                                                            (2, "generate", "no IK")]
    assert sorted(used(r)) == [1, 4, 6]


def test_exhausted_cell_raises_and_names_the_cell():
    cfgs = [cfg(0, (0,)), cfg(1, (1,)), cfg(2, (0,)), cfg(3, (1,))]
    r = rec()
    with pytest.raises(PoolExhausted, match=r"1/2 accepted, 2 rejected, no unused config left in cells \[\[1\]\]"):
        run(cfgs, 2, fail={1, 3}, r=r)
    assert used(r) == [0] and len(r["rejected"]) == 2


def test_exhaustion_when_the_pool_is_all_bad():
    with pytest.raises(PoolExhausted, match="0/2 accepted, 4 rejected"):
        run([cfg(i) for i in range(4)], 2, fail=set(range(4)))


def test_resume_after_a_kill_attempts_only_undecided_seeds():
    cfgs, calls = two_cells(), []

    class Kill(Exception):
        pass

    r = rec()

    def attempt(c):
        if len(r["used"]) + len(r["rejected"]) == 3:
            raise Kill
        return ("generate", Result.fail("bad")) if c.seed == 1 else ("generate", Result())
    with pytest.raises(Kill):
        walk(cfgs, r, 4, attempt, lambda: None)
    assert used(r) == [0, 2] and [e["seed"] for e in r["rejected"]] == [1]
    run(cfgs, 4, fail={1}, r=r, calls=calls)
    assert calls == [3, 5] and sorted(used(r)) == [0, 2, 3, 5]
    again = []
    run(cfgs, 4, fail={1}, r=r, calls=again)
    assert again == []                                      # a finished walk is a no-op


def test_resume_rebuilds_replacements_from_rejected_seeds():
    cfgs = two_cells()
    full = run(cfgs, 4, fail={1, 5})
    half = rec()
    half["used"], half["rejected"] = full["used"][:2], full["rejected"][:1]
    calls = []
    run(cfgs, 4, fail={1, 5}, r=half, calls=calls)
    assert 1 not in calls and calls[0] == 3
    assert sorted(used(half)) == sorted(used(full)) and len(half["rejected"]) == 2
