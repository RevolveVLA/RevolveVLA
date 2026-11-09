"""Dataset tests. test_dataset_unit.py: labels, checks, progress parser on stub data (no docker).
test_dataset.py: real conversions in revolve-dataset.

Fixtures are NOT committed binaries. The datagen episode is the recovery tests' cached one
(components/recovery/tests/conftest.py demo(): fridge seed 0 in runs/test_cache/recovery/datagen/, generated
there on its first run); the recovery episode is recovered from that episode's open start by
components/recovery/recover.py into runs/test_cache/dataset/recovery/ on the first run and reused after it
(delete that dir to regenerate). Cold cache: needs `./revolve build recovery` (+ datagen, simulator), the
kitchen assets and a free GPU, ~2-5 min; warm: CPU only. Outputs: runs/dataset_tests/<stamp>/ (git-ignored);
numbers.json there sums up the key values.

Needs: `./revolve build dataset`.
"""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

COMP = Path(__file__).resolve().parents[1]
REPO = COMP.parents[1]
sys.path[:0] = [str(COMP), str(COMP / "src" / "dataset")]
CACHE = REPO / "runs" / "test_cache" / "dataset"
RUN = REPO / "runs" / "dataset_tests" / time.strftime("%Y%m%d_%H%M%S")
TASK = "open_fridge_door"
INSPECT = COMP / "tests" / "inspect_dataset.py"


def load(p):
    return json.loads(Path(p).read_text())


def _recovery_conftest():
    s = importlib.util.spec_from_file_location("recovery_conftest", COMP.parent / "recovery" / "tests" / "conftest.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def preflight():
    import launch as D
    missing = [D.IMAGE] if subprocess.run(["docker", "image", "inspect", D.IMAGE], capture_output=True).returncode else []
    missing += [] if (D.INSTALL / "BUILD_OK").is_file() else [str(D.INSTALL / "BUILD_OK")]
    if missing:
        pytest.fail(f"not built: {missing} (./revolve build dataset)", pytrace=False)


def recovery_episode(RC, demo):
    """One recovery episode recovered from the demo's open start (cached in CACHE/recovery)."""
    import recover as R
    out = CACHE / "recovery"
    ep = out / R.key(demo, "open") / "episodes" / "ep_0"
    if not ((ep / "bag" / "metadata.yaml").is_file() and load(ep / "episode.json")["recovery"]["from_rollout"] == R.rel(demo)):
        RC.preflight()
        m = R.recover(TASK, [(demo, "open")], out, episodes_per_failure=1, workers=1)
        assert (ep / "bag" / "metadata.yaml").is_file(), m
    return ep


@pytest.fixture(scope="session")
def episodes():
    """{"demo": datagen episode dir, "recovery": recovery episode dir} (host paths)."""
    preflight()
    t0 = time.time()
    RC = _recovery_conftest()
    demo = CACHE.parent / "recovery" / "datagen" / TASK / "episodes" / "seed_0_a0"
    if not (demo / "bag" / "metadata.yaml").is_file():
        RC.preflight()
        demo = RC.demo(TASK)
    return dict(demo=demo, recovery=recovery_episode(RC, demo), wall_s=round(time.time() - t0, 1))


def in_image(*argv, timeout=600):
    """inspect_dataset.py (or any command) in revolve-dataset -> its last stdout line as JSON."""
    import launch as D
    code, out = D.in_image(["python3", INSPECT, *argv], timeout=timeout)
    assert code == 0, out[-3000:]
    return json.loads(out.strip().splitlines()[-1])


@pytest.fixture(scope="session")
def numbers():
    d = {}
    yield d
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "numbers.json").write_text(json.dumps(d, indent=1, default=str))
