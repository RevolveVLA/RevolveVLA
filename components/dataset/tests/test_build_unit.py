"""components/dataset/build.sh's Rosetta patch step on the host: its `for p in /src/patches/*.patch` loop, taken from
the script and run with bash and git against a tmp patches dir and a tmp repo (no docker)."""
import re
import shutil
import subprocess

import pytest

from conftest import COMP

pytestmark = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("git")), reason="needs bash and git")

LOOP_START = "for p in /src/patches/*.patch"
GOOD = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-hello\n+world\n"
BAD = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-goodbye\n+world\n"


def patch_loop() -> str:
    """build.sh's patch-applying loop, from its `for` up to the matching `done`."""
    text = (COMP / "build.sh").read_text()
    start = text.index(LOOP_START)
    depth = 0
    for m in re.finditer(r"\b(for|while|until|done)\b", text[start:]):
        depth += -1 if m.group(1) == "done" else 1
        if depth == 0:
            return text[start:start + m.end()]
    raise AssertionError("build.sh: the patch loop has no matching `done`")


def run_loop(tmp_path, patch_text):
    patches, repo = tmp_path / "patches", tmp_path / "rosetta"
    patches.mkdir()
    (patches / "0001-change.patch").write_text(patch_text)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.txt").write_text("hello\n")
    loop = patch_loop().replace("/src/patches", str(patches)).replace("/tmp/rosetta", str(repo))
    # in build.sh the colcon build follows the loop: a failure must stop the script there, not only set its status
    r = subprocess.run(["bash", "-c", f"set -eo pipefail; {loop}\necho after-the-loop"], capture_output=True, text=True)
    return r, repo


def test_a_patch_that_does_not_apply_fails_the_build(tmp_path):
    # a patch whose context is not in the repo makes the loop exit non-zero under set -e
    r, repo = run_loop(tmp_path, BAD)
    assert r.returncode != 0 and "after-the-loop" not in r.stdout, (r.stdout, r.stderr)
    assert (repo / "a.txt").read_text() == "hello\n"


def test_a_patch_that_applies_changes_the_repo(tmp_path):
    # a patch that applies exits 0 and its change is in the repo
    r, repo = run_loop(tmp_path, GOOD)
    assert r.returncode == 0 and "after-the-loop" in r.stdout, (r.stdout, r.stderr)
    assert (repo / "a.txt").read_text() == "world\n"
