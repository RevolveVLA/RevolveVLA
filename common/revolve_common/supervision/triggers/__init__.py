"""Trigger plugins: `@register("name")` on a class with `Spec` (a core.TriggerSpec), `__init__(params, task, *,
arm_joints, path)`, `phases`, `reset()`, `check(tick)` (`name` is set by `register`).
`get` imports `revolve_common.supervision.triggers.<name>`, else the module `<name>` from the dirs of
$REVOLVE_TRIGGER_PATH (external plugins); the module must register itself.
(Not datagen's Registry: common cannot depend on datagen.)"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys

from ...specs import SpecError

PLUGINS: dict[str, type] = {}
PATH_ENV = "REVOLVE_TRIGGER_PATH"


def register(name: str):
    def deco(cls):
        if PLUGINS.setdefault(name, cls) is not cls:
            raise SpecError(f"trigger {name!r} is already registered by {PLUGINS[name]}")
        cls.name = name
        return cls
    return deco


def _exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ImportError:
        return False


def get(name: str, path: str = "trigger") -> type:
    if name not in PLUGINS:
        sys.path[:0] = [p for p in os.environ.get(PATH_ENV, "").split(os.pathsep) if p and p not in sys.path]
        for module in (f"{__name__}.{name}", name):
            if _exists(module):
                importlib.import_module(module)
                if name in PLUGINS:
                    break
    if name not in PLUGINS:
        raise SpecError(f"{path}.plugin: no trigger {name!r} (registered: {', '.join(sorted(PLUGINS))}; "
                        f"external ones: modules in ${PATH_ENV})")
    return PLUGINS[name]
