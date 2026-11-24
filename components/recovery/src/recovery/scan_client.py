"""Scan entry, run in the simulator image (rosbag2_py): `python3 -m recovery.scan_client --spec <spec json> --run <eval run>
--out <scan.json>` writes the failures of the eval run (core/failures.py) as JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bag_io import read_messages
from .core.scan import scan_run
from .core.spec import parse


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="recovery.scan_client")
    ap.add_argument("--spec", required=True, help="the recovery spec as JSON (RecoverySpec.raw)")
    ap.add_argument("--run", required=True, help="the evaluation run's out")
    ap.add_argument("--out", required=True, help="the scan's JSON file")
    ap.add_argument("--only", action="append", help="<set>/<rollout key> to scan (repeatable; default: every rollout)")
    a = ap.parse_args(argv)
    only = {tuple(o.split("/", 1)) for o in a.only} if a.only else None
    scan = scan_run(parse(json.loads(a.spec), "<spec>"), Path(a.run), read_messages, only)
    Path(a.out).write_text(json.dumps(scan.to_dict(), indent=1))


if __name__ == "__main__":
    main()
