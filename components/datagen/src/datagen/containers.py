"""Stop the docker jobs of a run (host side, docker SDK imported when needed): the containers whose wrapper `--job-dir`
lies under the run's `out`, by that host path or by its /repo/... mapping. What an interrupted run leaves behind."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

LABEL = "revolve.component"


def stop_jobs(out: Path, log=print) -> int:
    """Kill the running jobs of the run at `out` -> how many. Never raises: a missing docker daemon means no jobs."""
    out = Path(out).resolve()
    prefixes = [str(out)] + (["/repo/" + str(out.relative_to(REPO))] if out.is_relative_to(REPO) else [])
    stopped = 0
    try:
        import docker
        for c in docker.from_env().containers.list(filters={"label": LABEL}):
            cmd = c.attrs["Config"]["Cmd"] or []
            job_dir = cmd[cmd.index("--job-dir") + 1] if "--job-dir" in cmd else ""
            if any(job_dir == p or job_dir.startswith(p + "/") for p in prefixes):
                c.kill()
                stopped += 1
    except Exception as e:                  # a job that finished meanwhile, no daemon: nothing (more) to stop
        log(f"stopping the jobs: {e!r}")
    return stopped
