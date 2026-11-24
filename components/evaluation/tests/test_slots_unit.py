"""core/slots.py: the worker slots (CPU, plain callables): order of commits, items before follow-up tasks, idle calls, failures."""
import threading
import time

import pytest

from evaluation.core.slots import Slots


def run(workers, items, call=lambda x: x * 10, commit=None, tasks=(), **kw):
    seen = []
    Slots(workers, **kw).run(items, call, commit or (lambda x, r: seen.append((x, r)) or ()), tasks)
    return seen


@pytest.mark.parametrize("workers", [1, 3, 8])
def test_results_are_committed_in_item_order_whatever_the_workers(workers):
    delays = {1: 0.05, 2: 0.0, 3: 0.02}
    seen = run(workers, [1, 2, 3, 4], call=lambda x: (time.sleep(delays.get(x, 0)), x * 10)[1])
    assert seen == [(1, 10), (2, 20), (3, 30), (4, 40)]


def test_at_most_the_workers_run_at_a_time():
    live, peak, lock = [0], [0], threading.Lock()

    def call(x):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.02)
        with lock:
            live[0] -= 1
    run(3, list(range(9)), call=call)
    assert peak[0] == 3


def test_a_slot_runs_a_follow_up_task_only_when_no_item_is_pending():
    order, lock = [], threading.Lock()

    def call(x):
        with lock:
            order.append(f"item{x}")

    def task(x):
        def t():
            with lock:
                order.append(f"task{x}")
        return t
    run(1, [1, 2, 3], call=call, commit=lambda x, r: [task(x)], tasks=[task(0)])
    assert order[:3] == ["item1", "item2", "item3"] and sorted(order[3:]) == ["task0", "task1", "task2", "task3"]
    assert order[3] == "task0"                                     # queued before the commits' tasks: first in line


def test_tasks_run_on_the_same_slots_and_the_run_waits_for_them():
    ends = []
    run(2, [1, 2], commit=lambda x, r: [lambda: (time.sleep(0.05), ends.append(x))], idle=lambda: None)
    assert sorted(ends) == [1, 2]


def test_idle_is_called_on_each_slot_when_it_has_no_item_left_and_when_it_ends():
    calls = []
    run(3, [1, 2], idle=lambda: calls.append(threading.current_thread().name))
    assert set(calls) == {"slot-0", "slot-1", "slot-2"}


def test_an_item_error_is_raised_after_the_earlier_commits_and_aborts():
    aborted, seen = [], []

    def call(x):
        if x == 3:
            raise ValueError("boom")
        return x
    with pytest.raises(ValueError, match="boom"):
        Slots(2, abort=lambda: aborted.append(1)).run([1, 2, 3, 4], call, lambda x, r: seen.append(x) or (), ())
    assert seen == [1, 2] and aborted == [1]


def test_a_task_error_stops_the_run_and_aborts():
    aborted, started = [], []

    def bad():
        raise RuntimeError("job died")
    with pytest.raises(RuntimeError, match="job died"):
        Slots(1, abort=lambda: aborted.append(1)).run([1], lambda x: x, lambda x, r: [bad, lambda: started.append(1)], ())
    assert aborted == [1] and started == []


def test_an_interrupt_of_the_commit_aborts_and_awaits_the_running_items():
    done = []

    def commit(x, r):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        Slots(2, abort=lambda: done.append("abort")).run([1, 2], lambda x: (time.sleep(0.05), done.append(x))[1], commit, ())
    assert "abort" in done and {1, 2} <= set(done)                  # both were running: awaited


def test_no_items_runs_only_the_given_tasks():
    ran = []
    run(2, [], tasks=[lambda: ran.append(1), lambda: ran.append(2)])
    assert sorted(ran) == [1, 2]


def test_a_slot_frees_what_it_kept_for_the_items_before_its_first_task():
    order = []
    run(1, [1, 2], call=lambda x: order.append(f"item{x}"), commit=lambda x, r: [lambda: order.append(f"task{x}")],
        idle=lambda: order.append("idle"))
    assert order[:3] == ["item1", "item2", "idle"] and order[3:5] == ["task1", "task2"]
