"""The params of a source or output plugin, checked with errors that name the yaml path (datagen's spec helpers)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from datagen.core.spec import _fail, _mapping, _str

from .files import resolve


class Params:
    def __init__(self, at: str, params: Mapping[str, Any], allowed: Sequence[str] = ()):
        self.at, self.params = at, _mapping(dict(params), at, optional=tuple(allowed))

    def path(self, key: str, root) -> str:
        if key not in self.params:
            _fail(self.at, f"missing key {key!r}")
        return resolve(root, _str(self.params[key], f"{self.at}.{key}"))

    def choice(self, key: str, choices: Sequence[str], default: str | None):
        v = self.params.get(key, default)
        if v != default and v not in choices:
            _fail(f"{self.at}.{key}", f"expected one of {tuple(choices)}, got {v!r}")
        return v

    def passed(self) -> bool | None:
        """`passed:` true (default) | false | any: which episodes by the evaluator's verdict (None: any)."""
        v = self.params.get("passed", True)
        if v not in (True, False, "any"):
            _fail(f"{self.at}.passed", f"expected true, false or any, got {v!r}")
        return None if v == "any" else v
