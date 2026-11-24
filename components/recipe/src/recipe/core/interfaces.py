"""What the recipe core needs of its plugins (it knows no source or output by name)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from .items import Read


class Source(ABC):
    """Reads one finished run of some component (a datagen manifest, an evaluation run, a recovery run) into items. Created as
    `Source(name, params, root)`: `name` is the recipe's name for it (the yaml path of its errors), `params` its yaml block
    minus `plugin`, `root` what its relative paths are relative to. `kind` names the data it has (an Item's `kind`, or
    `recovery` for a source that holds several)."""
    kind: str

    @abstractmethod
    def read(self) -> Read: ...


class Output(ABC):
    """Writes a finished selection. Created as `Output(params)`."""
    @abstractmethod
    def write(self, out: Path, selection: Mapping[str, Any]) -> list[Path]:
        """Write `selection` (the dict of `core/plan.py` `Plan.to_dict`) under `out`; the files written."""
