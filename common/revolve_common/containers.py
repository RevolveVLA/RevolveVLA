"""docker SDK helpers for the host-side launchers."""
from __future__ import annotations

import uuid

LAUNCH = "revolve.launch"


def run(dk, image, command=None, **kw):
    """dk.containers.run(detach=True) that removes the container when docker created it but could not start it
    (containers.run = create + start: e.g. "nvml error: gpu requires reset" left a created container behind)."""
    import docker
    tag = uuid.uuid4().hex
    kw["labels"] = {**(kw.get("labels") or {}), LAUNCH: tag}
    kw.setdefault("detach", True)
    try:
        return dk.containers.run(image, command, **kw)
    except docker.errors.APIError:
        for c in dk.containers.list(all=True, filters={"label": f"{LAUNCH}={tag}"}):
            try:
                c.remove(force=True)
            except docker.errors.APIError:
                pass
        raise
