"""Names of the containers a datagen run starts."""
import hashlib


def container_prefix(out) -> str:
    """`dg-<hash of the run's out>`: every container of the run is named `<prefix>-...`, so an interrupt can remove them all."""
    return "dg-" + hashlib.sha1(str(out).encode()).hexdigest()[:6]
