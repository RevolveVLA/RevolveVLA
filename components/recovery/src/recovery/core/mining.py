"""Which tries of an evaluation run phase recovery may mine: the sets the live recorder would roll out (core/leakage.py: `from.sets`,
default every set that is not held-out) after the same leakage check. The failed tries of the other sets that must stay unseen
are counted (`held_out`), never scanned. One function for the standalone `collect` and the coupled evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Collection

from . import failures, leakage
from .failures import Scan
from .interfaces import Scanner
from .spec import RecoverySpec

HELD_OUT = "held_out"                   # in Scan.counts: failed tries of sets kept unseen
LABEL = {HELD_OUT: "held-out, not mined"}


def mine(spec: RecoverySpec, scanner: Scanner, out: Path, only: Collection[tuple[str, str]] | None = None) -> tuple[Scan, dict]:
    """(the failures of `only` (default: the whole run) in the sets to mine, {mined: [..], held_out: [..]} the selection made).
    SpecError when a set to mine overlaps a set that must stay unseen."""
    sets = leakage.of_run(Path(spec.eval_run))
    mined, reserved = leakage.sets_of(sets, spec.eval_sets, spec.heldout, required=False)
    leakage.check(sets, mined, reserved)
    keys, held = set(), 0
    for set_name, key, _, t in failures.tries(Path(spec.eval_run), only):
        if set_name in mined:
            keys.add((set_name, key))
        elif set_name in reserved:
            held += not t["record"].get("error") and not t["record"]["passed"]
    scan = scanner.scan(out, keys) if keys else Scan()          # no key would mean the whole run to a scanner
    if held:
        scan.counts[HELD_OUT] += held
    return scan, {"mined": mined, "held_out": reserved}
