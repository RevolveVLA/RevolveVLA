"""Progress parser for revolve_common.wrapper (--parser recovery_progress:parse): PROGRESS {json} lines
(recovery_entry.py, recover.py, regen_sim.py) and plan_from_state.py's own output (its report printed
with indent=1: datagen's parser; the per-attempt reach lines)."""
import re

from progress import parse as datagen_parse     # components/datagen/src/datagen

REACH = re.compile(r"^attempt (\d+): reach (ok|FAILED)")
REACH_N = re.compile(r'^ "reach_success": (\d+),?$')


def parse(line):
    m = REACH.match(line)
    if m:
        return {"stage": "reach", "reach_attempt": int(m.group(1)), "reach_ok": m.group(2) == "ok"}
    m = REACH_N.match(line)
    if m:
        return {"reach_success": int(m.group(1))}
    return datagen_parse(line)
