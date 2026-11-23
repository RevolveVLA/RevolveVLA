"""One base class so the CLI can report any datagen failure as a plain message."""


class DatagenError(Exception):
    pass
