import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPECS = REPO / "specs"


@pytest.fixture
def specs_copy(tmp_path):
    """A writable copy of specs/ (for broken-spec tests)."""
    dst = tmp_path / "specs"
    shutil.copytree(SPECS, dst)
    return dst
