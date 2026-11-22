"""assets_pipeline/make_assets.sh on the host, against a fake docker first on PATH that logs its argv: one image
build, then the script itself in that image with --out mounted at /out. The image is $REVOLVE_ASSETS_IMAGE (when
set and non-empty), else revolve-assets:latest, so a per-user tag never overwrites a shared one."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "assets_pipeline" / "make_assets.sh"

pytestmark = pytest.mark.skipif(Path("/.dockerenv").exists(), reason="in a container the script runs its in-container branch")

FAKE_DOCKER = f"""#!{sys.executable}
import json, os, sys
with open(os.environ["FAKE_DOCKER_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
sys.exit(int(os.environ["FAKE_DOCKER_RUN_RC"]) if sys.argv[1] == "run" else 0)
"""


def run(tmp_path, *args, image=None, run_rc=0):
    """The script with the fake docker -> (CompletedProcess, the docker argv lists)."""
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "bin" / "docker").write_text(FAKE_DOCKER)
    (tmp_path / "bin" / "docker").chmod(0o755)
    log = tmp_path / "docker.log"
    env = {k: v for k, v in os.environ.items() if k != "REVOLVE_ASSETS_IMAGE"}
    env.update(PATH=f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}", FAKE_DOCKER_LOG=str(log),
               FAKE_DOCKER_RUN_RC=str(run_rc), **({} if image is None else {"REVOLVE_ASSETS_IMAGE": image}))
    proc = subprocess.run(["bash", str(SCRIPT), *args], cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=60, check=False)
    return proc, [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize("image, used, offline", [
    (None, "revolve-assets:latest", []),
    ("", "revolve-assets:latest", ["--offline"]),
    ("revolve-assets:someone-ct", "revolve-assets:someone-ct", []),
    ("localhost:5000/team/revolve-assets:v2", "localhost:5000/team/revolve-assets:v2", ["--offline"]),
], ids=["unset", "empty", "user-tag", "registry-tag"])
def test_builds_and_runs_one_image(tmp_path, image, used, offline):
    out = tmp_path / "a" / "assets"                   # created on the host when missing
    proc, calls = run(tmp_path, "--out", out, *offline, image=image)
    assert proc.returncode == 0 and out.is_dir(), proc.stderr
    assert calls == [
        ["build", "-q", "-t", used, "assets_pipeline"],
        ["run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-v", f"{REPO}:/repo",
         "-v", f"{out}:/out", "-w", "/repo", used, "bash", "assets_pipeline/make_assets.sh", "--out", "/out",
         *offline],
    ]


def test_unknown_argument_exits_2_without_docker(tmp_path):
    proc, calls = run(tmp_path, "--out", tmp_path / "out", "--bogus")
    assert proc.returncode == 2 and "usage" in proc.stdout and calls == []


def test_exit_code_is_the_docker_runs(tmp_path):
    proc, calls = run(tmp_path, "--out", tmp_path / "out", run_rc=3)
    assert proc.returncode == 3 and [c[0] for c in calls] == ["build", "run"]
