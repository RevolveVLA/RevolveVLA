"""The walk over a split's sampled configs until `target` are accepted. The first `target` configs (pool order, which
the samplers keep balanced) are the picks; a rejected pick is replaced by the next unused config of its own cell
(any unused one for cell-less samplers). Rejections are recorded with their reason and every decision is saved, so a
killed run resumes where it stopped: the walk is a function of the sampled configs and the decided seeds."""
from __future__ import annotations

from collections import deque
from typing import Callable, Sequence

from .errors import DatagenError
from .interfaces import Config, Result


class PoolExhausted(DatagenError):
    pass


def walk(configs: Sequence[Config], rec: dict, target: int,
         attempt: Callable[[Config], tuple[str, Result]], save: Callable[[], None]):
    """`attempt(config)` -> (step, Result). `rec` is the split's manifest block: accepted go to rec["used"] as
    {config, generate}, rejected to rec["rejected"] as {seed, step, reason}; configs already in either are not
    attempted again (a rejected one still gets its replacement). Raises PoolExhausted when a rejected pick's cell
    has no unused config left and the target is not reached."""
    accepted = {e["config"]["seed"] for e in rec["used"]}
    rejected = {e["seed"] for e in rec["rejected"]}
    queue = deque(configs[:target])
    taken = {c.seed for c in queue}
    dry = []                                    # cells whose rejected pick had no replacement
    while queue:
        config = queue.popleft()
        if config.seed in accepted:
            continue
        if config.seed not in rejected:
            step, res = attempt(config)
            if res.ok:
                rec["used"].append({"config": config.to_dict(), "generate": res.to_dict()})
            else:
                rec["rejected"].append({"seed": config.seed, "step": step, "reason": res.reason})
            save()
            if res.ok:
                continue
        spare = next((c for c in configs if c.cell == config.cell and c.seed not in taken), None)
        if spare is None:
            dry.append(list(config.cell))
        else:
            taken.add(spare.seed)
            queue.append(spare)
    if len(rec["used"]) < target:
        raise PoolExhausted(f"{len(rec['used'])}/{target} accepted, {len(rec['rejected'])} rejected, "
                            f"no unused config left in cells {dry} of the {len(configs)} sampled")
