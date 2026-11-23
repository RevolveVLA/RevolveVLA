"""Plugin registry: (kind, name) -> class. Plugins register themselves on import; `get` imports lazily, so core
(and `./revolve datagen <spec> sample` on the host) never loads a plugin's heavy dependencies.

`Registry(kinds, package, core, path_env)` is one registry per component (datagen's below, evaluation's in its own
core). Lookup order for a name not yet registered: the kind's core module (`core`: kind -> module),
`<package>.plugins.<name>`, `<package>.plugins.<kind>s.<name>` (`policy` -> `policies`), then `<name>` itself as a
module on sys.path (external plugins; the CLI calls `extend_path()` to add the dirs of the `path_env` variable). The
imported module must register (kind, name). The module functions `register` / `get` / `create` are datagen's."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Callable, Mapping, Sequence

from .errors import DatagenError


class RegistryError(DatagenError, LookupError):
    pass


def _plural(kind: str) -> str:
    return kind[:-1] + "ies" if kind.endswith("y") else kind + "s"


def _exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ImportError:         # a parent package of `module` is missing
        return False


class Registry:
    def __init__(self, kinds: Sequence[str], package: str, core: Mapping[str, str] = {}, path_env: str | None = None):
        self.kinds, self.package, self.core, self.path_env = tuple(kinds), package, dict(core), path_env
        self._plugins: dict[tuple[str, str], type] = {}

    def register(self, kind: str, name: str) -> Callable[[type], type]:
        """Class decorator: `@registry.register("generator", "moveit")`."""
        if kind not in self.kinds:
            raise RegistryError(f"unknown plugin kind {kind!r} (kinds: {self.kinds})")

        def deco(cls):
            if self._plugins.setdefault((kind, name), cls) is not cls:
                raise RegistryError(f"{kind} {name!r} is already registered by {self._plugins[kind, name]}")
            return cls
        return deco

    def _candidates(self, kind: str, name: str):
        if kind in self.core:
            yield self.core[kind]
        yield f"{self.package}.plugins.{name}"
        yield f"{self.package}.plugins.{_plural(kind)}.{name}"
        yield name

    def get(self, kind: str, name: str) -> type:
        if (kind, name) not in self._plugins:
            for module in self._candidates(kind, name):
                if _exists(module):
                    importlib.import_module(module)
                    if (kind, name) in self._plugins:
                        break
        if (kind, name) not in self._plugins:
            known = sorted(n for k, n in self._plugins if k == kind)
            raise RegistryError(f"no {kind} plugin {name!r} (registered: {known}; module lookup: "
                                f"{' | '.join(self._candidates(kind, name))})")
        return self._plugins[kind, name]

    def create(self, kind: str, name: str, *args):
        return self.get(kind, name)(*args)

    def extend_path(self):
        """Put the dirs of the `path_env` variable (os.pathsep separated) on sys.path, for external plugin modules."""
        sys.path[:0] = [p for p in os.environ.get(self.path_env or "", "").split(os.pathsep) if p]


KINDS = ("sampler", "randomiser", "instancer", "generator", "runner", "converter", "validator", "check")
CORE = {"sampler": "datagen.core.samplers", "check": "datagen.core.checks"}      # built-in plugins
_DEFAULT = Registry(KINDS, "datagen", CORE, "DATAGEN_PLUGIN_PATH")
_PLUGINS = _DEFAULT._plugins
register, get, create = _DEFAULT.register, _DEFAULT.get, _DEFAULT.create
