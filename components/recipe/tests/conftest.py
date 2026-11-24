"""Recipe tests: CPU only, on the tiny fake run dirs in tests/data (a datagen run, an evaluation run, a recovery run)."""
import shutil
import sys
from pathlib import Path

import pytest

COMP = Path(__file__).resolve().parents[1]
REPO = COMP.parents[1]
sys.path[:0] = [str(REPO / "common"), str(COMP / "src"), str(REPO / "components" / "datagen" / "src"), str(COMP / "tests")]

DATA = COMP / "tests" / "data"


@pytest.fixture
def root(tmp_path) -> Path:
    """A copy of tests/data (the relative paths in its files are relative to this root)."""
    shutil.copytree(DATA, tmp_path / "root")
    return tmp_path / "root"
