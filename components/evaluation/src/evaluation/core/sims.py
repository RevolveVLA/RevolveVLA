"""The names of a run's sim containers: `<prefix>-<k>` (and `-client`), so an interrupted run can remove exactly its own."""
from __future__ import annotations

import hashlib

from .spec import EvalSpec


def sim_prefix(spec: EvalSpec) -> str:
    return "ev-" + hashlib.sha1(str(spec.out).encode()).hexdigest()[:6]
