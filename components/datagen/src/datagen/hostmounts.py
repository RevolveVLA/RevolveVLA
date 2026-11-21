"""Host paths outside the repo that the docker jobs must reach: an absolute spec `out` (e.g. on another disk).

The launchers (datagen, simulator, dataset launch.py) only know the repo: `cpath` maps a host path to /repo/... and only
runs/ is mounted. `add` records such a path; `apply`, called where a job launcher is loaded, then makes
  - `cpath` of the loaded launchers return a path under a recorded one unchanged, and
  - every container this process starts bind-mount the recorded paths read-write at the same path
(the launchers stay as they are; nothing is recorded, so nothing is patched, for a run under the repo's runs/).
`out` is resolved first, so a symlink to another disk is that disk's path everywhere, resume included."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
LAUNCHERS = ("launch", "sim_launch", "dataset_launch")       # module names of the three launchers once loaded
MOUNTS: list[Path] = []


def add(path) -> None:
    """Record `path` (resolved) unless it lies under the repo's runs/."""
    p = Path(path).resolve()
    if not p.is_relative_to(REPO / "runs") and p not in MOUNTS:
        MOUNTS.append(p)


def clear() -> None:
    MOUNTS.clear()


def _under_mount(p: Path) -> bool:
    return any(p == m or m in p.parents for m in MOUNTS)


def _mapped(cpath):
    def mapped(p):
        q = Path(p).resolve()
        return str(q) if _under_mount(q) else cpath(p)
    mapped.hostmounts = True
    return mapped


def _with_mounts(run):
    def mounted(self, image, command=None, **kw):
        vols = kw.get("volumes")
        if isinstance(vols, dict):
            kw["volumes"] = {**vols, **{str(m): {"bind": str(m), "mode": "rw"} for m in MOUNTS}}
        return run(self, image, command, **kw)
    mounted.hostmounts = True
    return mounted


def apply() -> None:
    """Make the loaded launchers and the docker SDK honour the recorded paths (idempotent; no-op without any)."""
    if not MOUNTS:
        return
    from docker.models.containers import ContainerCollection
    for m in MOUNTS:
        m.mkdir(parents=True, exist_ok=True)        # docker would create a missing bind source as root
    if not getattr(ContainerCollection.run, "hostmounts", False):
        ContainerCollection.run = _with_mounts(ContainerCollection.run)
    for name in LAUNCHERS:
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "cpath") and not getattr(mod.cpath, "hostmounts", False):
            mod.cpath = _mapped(mod.cpath)
