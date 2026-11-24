"""Names of the containers a recovery run starts: `rc-<hash of the run's out>-...`, so an interrupt removes exactly its own."""
import hashlib


def prefix(spec) -> str:
    return "rc-" + hashlib.sha1(str(spec.out).encode()).hexdigest()[:6]
