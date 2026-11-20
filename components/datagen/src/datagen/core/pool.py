"""The walk over a split's sampled configs until `target` are accepted; rejections are recorded with their reason
and every decision is saved, so a killed run resumes where it stopped (PRD 02 adds in-cell replacement)."""
from __future__ import annotations

from typing import Callable, Sequence

from .errors import DatagenError
from .interfaces import Config, Result


class PoolExhausted(DatagenError):
    pass


def walk(configs: Sequence[Config], rec: dict, target: int,
         attempt: Callable[[Config], tuple[str, Result]], save: Callable[[], None]):
    """`attempt(config)` -> (step, Result). `rec` is the split's manifest block: accepted go to rec["used"] as
    {config, generate}, rejected to rec["rejected"] as {seed, step, reason}; configs already in either are skipped."""
    decided = {e["config"]["seed"] for e in rec["used"]} | {e["seed"] for e in rec["rejected"]}
    for config in configs:
        if len(rec["used"]) >= target:
            return
        if config.seed in decided:
            continue
        step, res = attempt(config)
        if res.ok:
            rec["used"].append({"config": config.to_dict(), "generate": res.to_dict()})
        else:
            rec["rejected"].append({"seed": config.seed, "step": step, "reason": res.reason})
        save()
    if len(rec["used"]) < target:
        raise PoolExhausted(f"{len(rec['used'])}/{target} accepted, {len(rec['rejected'])} rejected, "
                            f"pool of {len(configs)} configs used up")
