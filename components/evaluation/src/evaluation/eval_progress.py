"""Progress parser for revolve_common.wrapper (--parser eval_progress:parse) around evaluate.py: its PROGRESS {json}
lines (set, rollouts, rollouts_done, passed, first_try_passed, running), plus the per-rollout result lines
(`<seed>_r<i>: try 0: PASS, ...`) as last_rollout."""
import re

from revolve_common.wrapper import json_progress

ROLLOUT = re.compile(r" evaluate (\w+): (\S+_r\d+): (try 0: .*)$")


def parse(line):
    d = json_progress(line)
    if d is not None:
        return d
    m = ROLLOUT.search(line)
    return {"last_rollout": f"{m.group(2)}: {m.group(3)}"} if m else None
