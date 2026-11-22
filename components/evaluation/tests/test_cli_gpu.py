"""GPU system test (EVAL_GPU=1, built images, a free GPU): `./revolve evaluate` on 1 held-out fridge config with the oracle policy
(the expert's own actions, no checkpoint): one rollout, C0 (the phase verdict) and C2 (the state log) both pass. Output:
runs/evaluation_tests/<stamp>_cli_gpu/ (git-ignored)."""
import json
import subprocess
import sys
import time

import pytest
import yaml

from conftest import REPO

pytestmark = pytest.mark.gpu


def test_oracle_on_one_heldout_config_passes_c0_and_c2():
    out = REPO / "runs" / "evaluation_tests" / f"{time.strftime('%Y%m%d_%H%M%S')}_cli_gpu"
    spec = yaml.safe_load((REPO / "specs" / "eval" / "fridge_smoke_eval.yaml").read_text()) | {"out": str(out.relative_to(REPO))}
    out.mkdir(parents=True)
    (out / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    r = subprocess.run([sys.executable, str(REPO / "revolve"), "evaluate", str(out / "spec.yaml"), "all"], cwd=REPO,
                       capture_output=True, text=True, timeout=3600)
    (out / "evaluate.log").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((REPO / spec["out"] / "results.json").read_text())
    (entry,) = res["sets"]["heldout"]["rollouts"].values()
    (first,) = entry["tries"]
    assert first["record"]["passed"] and not first["record"].get("error")           # C0: every phase passed
    assert entry["passed"] and first["detail"]["criterion"] == "c2"   # C2
    assert all(c["ok"] for c in res["sets"]["heldout"]["checks"].values())
