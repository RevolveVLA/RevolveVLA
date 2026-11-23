"""The walk over a split's sampled configs until `target` are accepted. The first `target` configs (pool order, which
the samplers keep balanced) are the picks; a rejected pick is replaced by the next unused config of its own cell
(any unused one for cell-less samplers). Rejections are recorded with their reason and every decision is saved, so a
killed run resumes where it stopped: the walk is a function of the sampled configs and the decided seeds.

`workers` > 1 runs the attempts of the next picks on that many threads at once, but every decision (accept, reject,
replace, save) is taken on the calling thread in queue order, so the manifest is the one a single worker would write."""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, Sequence

from .errors import DatagenError
from .interfaces import Config, Result


class PoolExhausted(DatagenError):
    pass


def walk(configs: Sequence[Config], rec: dict, target: int,
         attempt: Callable[[Config], tuple[str, Result]], save: Callable[[], None], workers: int = 1,
         accept: Callable[[Config, str, Result], tuple[str, Result]] | None = None,
         abort: Callable[[], None] | None = None, discard: Callable[[Sequence[Config]], None] | None = None):
    """`attempt(config)` -> (step, Result), on up to `workers` threads (it must not touch `rec`). `accept(config, step,
    result)` -> (step, result) is the decision hook, run in queue order on this thread (e.g. the realised min_distance,
    which reads the accepted ones). `rec` is the split's manifest block: accepted go to rec["used"] as {config,
    generate}, rejected to rec["rejected"] as {seed, step, reason}; configs already in either are not attempted again
    (a rejected one still gets its replacement). Raises PoolExhausted when a rejected pick's cell has no unused config
    left and the target is not reached. On any exception (an interrupt too) the queued attempts are cancelled,
    `abort()` is called (stop what the running ones started), they are awaited and `discard(configs)` gets those that
    had not finished, whose partial output must not be resumed from."""
    accepted = {e["config"]["seed"] for e in rec["used"]}
    rejected = {e["seed"] for e in rec["rejected"]}
    queue = deque(configs[:target])
    taken = {c.seed for c in queue}
    dry = []                                    # cells whose rejected pick had no replacement
    flying = {}                                 # seed -> future of its attempt
    ex = ThreadPoolExecutor(max_workers=workers)

    def pending():
        return [f for f in flying.values() if not f.done()]

    try:
        while queue:
            for c in queue:                     # start the next undecided picks, in queue order
                if len(pending()) >= workers:
                    break
                if c.seed not in accepted | rejected and c.seed not in flying:
                    flying[c.seed] = ex.submit(attempt, c)
            config = queue[0]
            if config.seed in accepted:
                queue.popleft()
                continue
            if config.seed not in rejected:
                if config.seed not in flying or not flying[config.seed].done():
                    wait(pending(), return_when=FIRST_COMPLETED)
                    continue
                queue.popleft()
                step, res = flying.pop(config.seed).result()
                step, res = accept(config, step, res) if accept else (step, res)
                if res.ok:
                    rec["used"].append({"config": config.to_dict(), "generate": res.to_dict()})
                else:
                    rec["rejected"].append({"seed": config.seed, "step": step, "reason": res.reason})
                save()
                if res.ok:
                    continue
            else:
                queue.popleft()
            spare = next((c for c in configs if c.cell == config.cell and c.seed not in taken), None)
            if spare is None:
                dry.append(list(config.cell))
            else:
                taken.add(spare.seed)
                queue.append(spare)
    except BaseException:
        left = [c for c in queue if c.seed in flying and not flying[c.seed].done()]
        ex.shutdown(wait=False, cancel_futures=True)
        if abort:
            abort()
        ex.shutdown(wait=True)
        if discard:
            discard(left)
        raise
    ex.shutdown()
    if len(rec["used"]) < target:
        raise PoolExhausted(f"{len(rec['used'])}/{target} accepted, {len(rec['rejected'])} rejected, "
                            f"no unused config left in cells {dry} of the {len(configs)} sampled")


def ordered(items: Sequence, call: Callable, commit: Callable, workers: int = 1, abort: Callable[[], None] | None = None):
    """`call(item)` on up to `workers` threads; `commit(item, result)` on this thread in `items` order. On any exception
    the queued calls are cancelled, `abort()` is called and the running ones are awaited."""
    ex = ThreadPoolExecutor(max_workers=workers)
    futures = [ex.submit(call, i) for i in items]
    try:
        for item, f in zip(items, futures):
            commit(item, f.result())
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        if abort:
            abort()
        ex.shutdown(wait=True)
        raise
    ex.shutdown()
