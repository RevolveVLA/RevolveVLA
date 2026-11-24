"""Recovery failures are datagen's: one base class (and `SpecError`, `RegistryError`) so the CLI reports any as a message."""
from datagen.core.errors import DatagenError


class RecoveryError(DatagenError):
    pass


class JobError(RecoveryError):
    """A recovery job that cannot be run (not a failed episode): infrastructure retries used up, an unreadable scene."""
