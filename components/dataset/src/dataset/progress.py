"""Progress parser for revolve_common.wrapper (--parser progress:parse): the PROGRESS {json} lines of
entry.py / port.py (episodes_converted after every saved episode), plus rosetta_port's own log lines
("<i> / <n> episodes processed" before episode i, "Completed: <ok>/<n> episodes (<failed> failed) ...")."""
import re

from revolve_common.wrapper import json_progress

PROCESSED = re.compile(r"^(\d+) / (\d+) episodes processed$")
COMPLETED = re.compile(r"^Completed: (\d+)/(\d+) episodes \((\d+) failed\)")


def parse(line):
    d = json_progress(line)
    if d is not None:
        return d
    m = PROCESSED.match(line.strip())
    if m:
        return {"episodes_converted": int(m.group(1)), "episodes_total": int(m.group(2))}
    m = COMPLETED.match(line.strip())
    if m:
        return {"episodes_converted": int(m.group(1)), "episodes_total": int(m.group(2)),
                "episodes_failed": int(m.group(3))}
    return None
