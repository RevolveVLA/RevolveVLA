"""Tests of code the new simulator made obsolete (DATAGEN-v2.md s13): skipped unless REVOLVE_LEGACY=1; PRD 07 deletes them.

A conftest lists them as LEGACY = {test file: None (the whole file, which may not even import) | {test function names}}
and wires `ignored(LEGACY)` into `collect_ignore` and `mark(items, LEGACY)` into pytest_collection_modifyitems."""
import os

import pytest

ON = os.environ.get("REVOLVE_LEGACY") == "1"


def ignored(legacy) -> list[str]:
    """The files that are not collected at all (collect_ignore)."""
    return [] if ON else [f for f, tests in legacy.items() if tests is None]


def mark(items, legacy) -> None:
    """Add the `legacy` marker to the listed tests (or any carrying it already) and skip them unless REVOLVE_LEGACY=1."""
    skip = pytest.mark.skip(reason="legacy: code the new simulator removed; REVOLVE_LEGACY=1 runs it")
    for item in items:
        tests = legacy.get(item.path.name, ())
        if tests is None or item.originalname in tests:
            item.add_marker(pytest.mark.legacy)
        if not ON and "legacy" in item.keywords:
            item.add_marker(skip)
