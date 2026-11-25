"""What the dataset component reads from a robot's Rosetta contract (specs/robots/<r>/contract.yaml), without Rosetta:
the topics a bag must carry and the features the dataset must have. Pure Python (unit-tested on the host); Rosetta
itself decodes the bags with the same file."""
from __future__ import annotations

from pathlib import Path

import yaml

IMAGE_PREFIX = "observation.images."
KEY_SECTIONS = ("observations", "actions", "rewards", "signals", "info", "complementary_data")   # keyed features


def load(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def topics(contract: dict) -> list[str]:
    """Every channel topic the contract names (observations, actions, tasks, any other section), in file order."""
    out = []

    def walk(v):
        if isinstance(v, dict):
            ch = v.get("channel")
            if isinstance(ch, dict) and isinstance(ch.get("topic"), str) and ch["topic"] not in out:
                out.append(ch["topic"])
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    for k, v in contract.items():
        if not k.startswith("x-"):           # YAML anchors only: their contents are merged where they are used
            walk(v)
    return out


def features(contract: dict) -> dict:
    """{key: shape} of the contract's keyed features: images (h, w, 3) from their resize operator, vectors the
    concatenated selects of the key's sources (Rosetta concatenates multi-source keys in list order)."""
    out = {}
    for section in KEY_SECTIONS:
        for key, spec in (contract.get(section) or {}).items():
            srcs = spec if isinstance(spec, list) else [spec]
            size = next((op["resize"] for s in srcs for op in s.get("apply", []) or []
                         if isinstance(op, dict) and "resize" in op), None)
            out[key] = (int(size[0]), int(size[1]), 3) if size else (sum(len(s["select"]) for s in srcs),)
    return out
