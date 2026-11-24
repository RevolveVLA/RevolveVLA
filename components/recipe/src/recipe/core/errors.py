"""Recipe failures are datagen's: one base class (and `SpecError`, `RegistryError`) so the CLI reports any as a message."""
from datagen.core.errors import DatagenError


class RecipeError(DatagenError):
    """Data the recipe cannot use: an unreadable run, a missing exclude path, frames wanted of an episode with no duration."""
