"""Plugin registry: (kind, name) -> class. Plugins register themselves on import; `get` imports lazily, so core
(and `./revolve datagen <spec> sample` on the host) never loads a plugin's heavy dependencies.

Lookup order for a name not yet registered: the kind's core module (CORE), `datagen.plugins.<name>`,
`datagen.plugins.<kind>s.<name>`, then `<name>` itself as a module on sys.path (external plugins; the CLI adds
$DATAGEN_PLUGIN_PATH). The imported module must register (kind, name)."""
from __future__ import annotations

import importlib
import importlib.util
from typing import Callable

from .errors import DatagenError

KINDS = ("sampler", "randomiser", "generator", "recorder", "converter", "validator", "check")
CORE = {"sampler": "datagen.core.samplers", "check": "datagen.core.checks"}      # built-in plugins
_PLUGINS: dict[tuple[str, str], type] = {}


class RegistryError(DatagenError, LookupError):
    pass


def register(kind: str, name: str) -> Callable[[type], type]:
    """Class decorator: `@register("generator", "automoma")`."""
    if kind not in KINDS:
        raise RegistryError(f"unknown plugin kind {kind!r} (kinds: {KINDS})")

    def deco(cls):
        if _PLUGINS.setdefault((kind, name), cls) is not cls:
            raise RegistryError(f"{kind} {name!r} is already registered by {_PLUGINS[kind, name]}")
        return cls
    return deco


def _candidates(kind: str, name: str):
    if kind in CORE:
        yield CORE[kind]
    yield f"datagen.plugins.{name}"
    yield f"datagen.plugins.{kind}s.{name}"
    yield name


def _exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ImportError:         # a parent package of `module` is missing
        return False


def get(kind: str, name: str) -> type:
    if (kind, name) not in _PLUGINS:
        for module in _candidates(kind, name):
            if _exists(module):
                importlib.import_module(module)
                if (kind, name) in _PLUGINS:
                    break
    if (kind, name) not in _PLUGINS:
        known = sorted(n for k, n in _PLUGINS if k == kind)
        raise RegistryError(f"no {kind} plugin {name!r} (registered: {known}; module lookup: "
                            f"{' | '.join(_candidates(kind, name))})")
    return _PLUGINS[kind, name]


def create(kind: str, name: str, *args):
    return get(kind, name)(*args)
