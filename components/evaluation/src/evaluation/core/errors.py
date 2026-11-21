"""Evaluation failures are datagen's: one base class (and `SpecError`, `RegistryError`) so the CLI reports any as a message."""
from datagen.core.errors import DatagenError


class EvalError(DatagenError):
    pass
