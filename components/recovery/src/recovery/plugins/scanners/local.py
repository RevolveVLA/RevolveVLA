"""scanner local: the scan in this process. Needs the bag reader, i.e. a ROS 2 environment (rosbag2_py); `reader`
(`module:function`, default the real one) names another one.

Params: `reader`."""
from __future__ import annotations

import importlib
from pathlib import Path

from ...core.interfaces import Scanner
from ...core.registry import REGISTRY
from ...core.scan import scan_run

REAL = "recovery.bag_io:read_messages"


@REGISTRY.register("scanner", "local")
class LocalScanner(Scanner):
    def __init__(self, spec, params):
        self.spec, self.reader = spec, params.get("reader", REAL)

    def scan(self, out: Path, only=None):
        module, _, name = self.reader.partition(":")
        return scan_run(self.spec, Path(self.spec.eval_run), getattr(importlib.import_module(module), name), only)
