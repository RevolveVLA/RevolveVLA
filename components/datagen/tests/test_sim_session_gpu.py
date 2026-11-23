"""GPU + docker test of SimSession (revolve_common.sim_session): fridge / mobile_panda, ready, two episodes with a reset
and no commands (both fail on their timeout), the files complete, teardown leaves no container. Skipped unless
DATAGEN_GPU=1; needs `./revolve build simulator` + assets. Output runs/session_test_<pid>/ (removed when it passes)."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from revolve_common.sim_session import DomainPool, RelaunchPolicy, SimSession, run_with_relaunch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("DATAGEN_GPU") != "1", reason="needs a GPU: DATAGEN_GPU=1")]

REPO = Path(__file__).resolve().parents[3]
CLIENT = "/repo/components/datagen/tests/sim_session_client.py"


def containers(prefix):
    r = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True)
    return [n for n in r.stdout.split() if n.startswith(prefix)]


def test_two_episodes_with_a_reset_then_teardown():
    out = REPO / "runs" / f"session_test_{os.getpid()}"
    prefix = f"sess-{os.getpid()}"
    name, domain = DomainPool(prefix).lease()
    policy = RelaunchPolicy()

    def work(session):
        c = session.client(["python3", CLIENT, str(out), "2"])
        c.run(out.parent / f"{out.name}_client.log", timeout=600, watch=session.check)
        line = [ln for ln in (out.parent / f"{out.name}_client.log").read_text().splitlines() if ln.startswith("RESULT ")][-1]
        return json.loads(line[len("RESULT "):])
    try:
        res = run_with_relaunch(SimSession("kitchen_partnet", "mobile_panda", "open_fridge_door", out, name, domain,
                                           "no-images", log=out.parent / f"{out.name}_sim.log"), work, policy)
        eps = res["episodes"]
        assert len(eps) == 2 and eps[1]["episode"] == eps[0]["episode"] + 1
        for e in eps:
            assert not e["result"]["passed"] and e["result"]["reason"] == "timeout_s" and e["files"] == e["result"]
            assert (out / f"episode_{e['episode']}.json").is_file() and (Path(e["result"]["bag"]) / "metadata.yaml").is_file()
        assert not containers(prefix)
    finally:
        subprocess.run(["docker", "rm", "-f", name, f"{name}-client"], capture_output=True)
    for p in (out, *out.parent.glob(f"{out.name}_*.log")):
        shutil.rmtree(p) if p.is_dir() else p.unlink()
