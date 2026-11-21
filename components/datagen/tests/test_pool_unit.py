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


# ------------------------------------------------------------------ parallel attempts
import threading
import time

from datagen.core.pool import ordered


def slow_attempt(fail=(), delays=None, seen=None):
    """attempt(c): a sleep of delays[seed] (later seeds finish first), then ok or failed; `seen` {now, peak}."""
    lock = threading.Lock()

    def attempt(c):
        with lock:
            seen["now"] += 1
            seen["peak"] = max(seen["peak"], seen["now"])
        time.sleep((delays or {}).get(c.seed, 0))
        with lock:
            seen["now"] -= 1
        return ("generate", Result.fail(f"bad {c.seed}")) if c.seed in fail else ("generate", Result())
    return attempt


def walked(workers, configs, target, fail, delays, accept=None):
    r, seen = rec(), {"now": 0, "peak": 0}
    walk(configs, r, target, slow_attempt(fail, delays, seen), lambda: None, workers, accept)
    return r, seen


def test_workers_change_nothing_but_the_timing():
    delays = {0: 0.15, 1: 0.1, 2: 0.0, 3: 0.12, 4: 0.0, 5: 0.05, 6: 0.0, 7: 0.0}   # later seeds finish first
    one, seen1 = walked(1, two_cells(), 6, {1, 4, 5}, delays)
    three, seen3 = walked(3, two_cells(), 6, {1, 4, 5}, delays)
    assert one == three and len(one["used"]) == 6 and [e["seed"] for e in one["rejected"]] == [1, 4, 5]
    assert seen1["peak"] == 1 and 1 < seen3["peak"] <= 3                    # really parallel, never above `workers`


def test_the_accept_hook_decides_in_queue_order():
    def accept(c, step, res):                  # refuses seed 2 only when seed 0 and 1 are already accepted
        return ("realised", Result.fail("close")) if c.seed == 2 and len(used(seen_rec)) == 2 else (step, res)
    seen_rec = rec()
    walk([cfg(i) for i in range(8)], seen_rec, 3, slow_attempt(delays={0: 0.1, 1: 0.1}, seen={"now": 0, "peak": 0}),
         lambda: None, 3, accept)
    assert used(seen_rec) == [0, 1, 3] and seen_rec["rejected"] == [{"seed": 2, "step": "realised", "reason": "close"}]


def test_resume_with_workers_attempts_only_undecided_seeds():
    r = rec()
    r["used"] = [{"config": cfg(0).to_dict(), "generate": Result().to_dict()}]
    r["rejected"] = [{"seed": 1, "step": "generate", "reason": "bad"}]
    calls = []
    run([cfg(i) for i in range(6)], 3, r=r, calls=calls)
    assert sorted(calls) == [2, 3] and used(r) == [0, 2, 3]


def test_an_error_cancels_the_queue_aborts_waits_and_discards_the_unfinished():
    events, lock = [], threading.Lock()

    def attempt(c):
        if c.seed == 0:
            time.sleep(0.1)                    # seed 1 is under way when this one fails
            raise KeyboardInterrupt("stop")
        time.sleep(0.4 if c.seed == 1 else 0)
        with lock:
            events.append(f"done {c.seed}")
        return "generate", Result()
    left = []
    r = rec()
    with pytest.raises(KeyboardInterrupt):
        walk([cfg(i) for i in range(6)], r, 6, attempt, lambda: None, 2, None, lambda: events.append("abort"), left.extend)
    assert events[0] == "abort" and events[-1] == "done 1"           # aborted first, then the running one was awaited
    assert {1} <= {c.seed for c in left} <= {1, 2} and "done 4" not in events and r["used"] == []


def test_ordered_commits_in_item_order_whatever_the_finish_order():
    got, seen = [], {"now": 0, "peak": 0}
    attempt = slow_attempt(delays={0: 0.15, 1: 0.05}, seen=seen)
    ordered([cfg(0), cfg(1), cfg(2), cfg(3)], attempt, lambda c, res: got.append(c.seed), 3)
    assert got == [0, 1, 2, 3] and seen["peak"] > 1


def test_ordered_error_aborts_and_awaits():
    events = []

    def call(i):
        if i == 0:
            time.sleep(0.1)                    # item 1 is under way when this one fails
            raise ValueError("boom")
        time.sleep(0.2)
        events.append("done")
    with pytest.raises(ValueError):
        ordered([0, 1], call, lambda i, r: None, 2, lambda: events.append("abort"))
    assert events == ["abort", "done"]
