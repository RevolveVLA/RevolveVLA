"""scanner container: the scan in a container from the simulator image (rosbag2_py: the host has none), beside no sim:
`python3 -m recovery.scan_client` over the eval run's folder, its result in `<out>/scan.json`. Params: `wall_s` (1800)."""
from __future__ import annotations

import json
from pathlib import Path

from revolve_common.sim_session import ClientContainer

from ...core.errors import RecoveryError
from ...core.failures import Scan
from ...core.interfaces import Scanner
from ...core.naming import prefix
from ...core.registry import REGISTRY

DATAGEN_SRC, RECOVERY_SRC = "/repo/components/datagen/src", "/repo/components/recovery/src"    # in the container


@REGISTRY.register("scanner", "container")
class ContainerScanner(Scanner):
    def __init__(self, spec, params):
        self.spec, self.wall_s = spec, float(params.get("wall_s", 1800))

    def command(self, out: Path) -> list[str]:
        return ["python3", "-m", "recovery.scan_client", "--spec", json.dumps(self.spec.raw), "--run", self.spec.eval_run,
                "--out", str(out / "scan.json")]

    def scan(self, out: Path) -> Scan:
        container = ClientContainer(f"{prefix(self.spec)}-scan", 0, self.command(out), out,
                                    pythonpath=[DATAGEN_SRC, RECOVERY_SRC], mounts=[self.spec.eval_run])
        log = out / "logs" / "scan.log"
        try:
            rc = container.run(log, timeout=self.wall_s)
        except FileNotFoundError:
            raise RecoveryError("scan: docker is not available") from None
        if rc != 0:
            raise RecoveryError(f"scan: the container exited with {rc}; see {log}")
        return Scan.from_dict(json.loads((out / "scan.json").read_text()))
