"""Host side of the dataset component: its entrypoint (src/dataset/cli.py) in the dataset image, CPU only, offline.

  run(argv)                          ./revolve dataset <argv>: the CLI in a container, output on the terminal
  job = start(argv[, job_dir])       the same command under revolve_common.wrapper (job.json in; status.json,
  res = wait(job)                    result.json, stdout.log out; GET /status, GET /result, POST /cancel on :8000);
                                     job_dir defaults to <out's parent>/jobs/dataset_<kind>_<out's name>
  res = convert(manifests, out, turn, split=None)    start + wait, the manager's calls
  res = merge(datasets, out)

argv: `convert --manifest M ... [--split S] --turn K --out D` or `merge --out D DATASET...`, parsed with the
entrypoint's own parser (cli.parser(): no abbreviated options), so the mounts are exactly the paths it opens.
Mounts: the repo read-only at /repo, with runs/ writable; Rosetta's install (ThirdParty/rosetta/install) read-only at
/opt/rosetta/install, where it was built. The repo is also mounted at its own host path (runs/ writable there too),
because manifests and the simulator's results carry host paths; so do paths outside the repo: each manifest's
directory, each absolute bag or result path of a passed entry and each input dataset read-only, the output's parent
directory and the job directory writable, all at their own paths. The container runs as the calling user, in the
caller's directory when that is inside the repo. res: {exit, wall_s, result (result.json), dir (the job dir)}."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "components" / "dataset" / "src"
IMAGE = os.environ.get("REVOLVE_DATASET_IMAGE", "revolve-dataset:latest")
INSTALL = REPO / "ThirdParty" / "rosetta" / "install"
ENTRY = ["python3", "-m", "dataset.cli"]
LABEL = "revolve.component=dataset"
ENV = {"HOME": "/tmp", "HF_HUB_OFFLINE": "1", "HF_HOME": "/tmp/hf", "HF_LEROBOT_HOME": "/tmp/hf/lerobot",
       "REVOLVE_REPO": "/repo", "PYTHONPATH": "/repo/common:/repo/components/dataset/src",
       "PYTHONDONTWRITEBYTECODE": "1"}
SOURCE = 'source /opt/rosetta/install/setup.bash && exec "$@"'


class LaunchError(RuntimeError):
    pass


def _dataset(name: str):
    """dataset.<name> from components/dataset/src, imported without leaving that directory on sys.path (the manager
    and datagen load this module as a library). Its imports are pure Python."""
    if str(SRC) in sys.path:
        return importlib.import_module(f"dataset.{name}")
    sys.path.insert(0, str(SRC))
    try:
        return importlib.import_module(f"dataset.{name}")
    finally:
        sys.path.remove(str(SRC))


def _under(p: Path, root: Path) -> bool:
    return p == root or p.is_relative_to(root)


def _parse(argv):
    """argv -> the entrypoint's parsed arguments, with its own parser (cli.parser()); LaunchError on a bad argv. --help
    prints and exits 0 (SystemExit) as the entrypoint would."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return _dataset("cli").parser().parse_args(list(argv))
    except SystemExit as e:
        if not e.code:
            raise
        msg = err.getvalue().strip().splitlines()
        raise LaunchError(msg[-1] if msg else f"bad arguments {list(argv)}") from None


def _paths(argv, cwd: Path):
    """-> (the canonical argv, with relative paths made absolute when cwd is outside the repo, read-only dirs,
    writable dirs). Parsed with the entrypoint's parser, so launch mounts exactly the paths the entrypoint opens."""
    if not argv:
        return [], [], []
    a = _parse(argv)
    in_repo = _under(cwd, REPO)

    def fix(x):
        p = Path(x)
        return (str(p) if p.is_absolute() or in_repo else str(cwd / p)), (cwd / p).resolve()

    out, out_p = fix(a.out)
    if _under(out_p, REPO) and not _under(out_p, REPO / "runs"):
        raise LaunchError(f"--out {out_p}: inside the repo only under runs/ (the repo is mounted read-only)")
    ro, rw = [], [out_p.parent]
    if a.cmd == "convert":
        M = _dataset("manifest")
        args = ["convert"]
        for m in a.manifest:
            arg, p = fix(m)
            args += ["--manifest", arg]
            ro.append(p.parent)
            plan = M.read([str(p)])
            ro += [q.parent for e in plan.episodes for q in (e.bag, e.result)]
        args += (["--split", a.split] if a.split is not None else []) + ["--turn", str(a.turn), "--out", out]
    else:
        args = ["merge", "--out", out, "--"]
        for d in a.datasets:
            arg, p = fix(d)
            args.append(arg)
            ro.append(p)
    return args, ro, rw


def _check_build():
    ok = INSTALL / "BUILD_OK"
    if not ok.is_file():
        raise LaunchError("Rosetta is not built: ./revolve build dataset")
    fields = ok.read_text().split()
    if len(fields) < 2 or fields[1] != "lyrical":
        raise LaunchError(f"Rosetta's install was not built for the dataset image (ROS 2 lyrical; {ok}: "
                          f"{' '.join(fields)!r}): ./revolve build dataset")


def docker_cmd(argv, cwd=None, job_dir=None) -> tuple[list[str], list[str]]:
    """-> (the `docker run` options: user, env, network, mounts; the entrypoint command with the canonical argv).
    job_dir: under the HTTP job wrapper (the job dir writable, the network up for its port 8000). The caller adds its
    mode (--rm, -d, --name) and runs `[*options, *mode, *container(command)]`."""
    _check_build()
    cwd = Path(cwd or os.getcwd()).resolve()
    args, ro, rw = _paths(argv, cwd)
    if job_dir is not None:
        rw.append(Path(job_dir).resolve())
    (REPO / "runs").mkdir(exist_ok=True)
    cmd = ["docker", "run", "--init", "-u", f"{os.getuid()}:{os.getgid()}", "--label", LABEL, "--shm-size", "1g",
           "-w", str(cwd if _under(cwd, REPO) else REPO), "--network", "bridge" if job_dir else "none"]
    for k, v in ENV.items():
        cmd += ["-e", f"{k}={v}"]
    mounts = [(REPO, "/repo", "ro"), (REPO / "runs", "/repo/runs", "rw"), (REPO, str(REPO), "ro"),
              (REPO / "runs", str(REPO / "runs"), "rw"), (INSTALL, "/opt/rosetta/install", "ro")]
    done = [REPO]
    for d in rw:
        if not _under(d, REPO):     # mounted at its own path: docker would create a missing one as root
            d.mkdir(parents=True, exist_ok=True)
    for d, mode in [(d, "rw") for d in rw] + [(d, "ro") for d in ro]:
        if not any(_under(d, x) for x in done) and d.exists():
            mounts.append((d, str(d), mode))
            done.append(d)
    for src, dst, mode in mounts:
        cmd += ["-v", f"{src}:{dst}:{mode}"]
    return cmd, [*ENTRY, *args]


def container(command) -> list[str]:
    """The image and a command to run in it, with Rosetta's install sourced."""
    return [IMAGE, "bash", "-c", SOURCE, "dataset", *command]


def run(argv, cwd=None) -> int:
    try:
        opts, entry = docker_cmd(argv, cwd)
    except LaunchError as e:
        print(f"revolve dataset: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:     # --help
        return int(e.code or 0)
    return subprocess.call([*opts, "--rm", *container(entry)])


@dataclass
class Job:
    container: str
    dir: Path
    t0: float


def default_job_dir(out, kind) -> Path:
    out = Path(out).resolve()
    return out.parent / "jobs" / f"dataset_{kind}_{out.name}"


def start(argv, job_dir=None, linger_s=5.0) -> Job:
    """The entrypoint under the job wrapper in a detached container. job_dir: writable in the container (under the
    repo's runs/, or anywhere outside the repo); default from argv's --out (default_job_dir)."""
    from revolve_common import job as J
    if job_dir is None:
        a = _parse(argv)
        job_dir = default_job_dir(a.out, a.cmd)
    job_dir = Path(job_dir).resolve()
    if _under(job_dir, REPO) and not _under(job_dir, REPO / "runs"):
        raise LaunchError(f"job dir {job_dir}: inside the repo only under runs/")
    job_dir.mkdir(parents=True, exist_ok=True)
    for f in (J.STATUS, J.RESULT, J.LOG):
        (job_dir / f).unlink(missing_ok=True)
    name = f"dataset_{job_dir.name}_{uuid.uuid4().hex[:6]}"
    opts, entry = docker_cmd(argv, job_dir=job_dir)
    wrapped = ["python3", "-m", "revolve_common.wrapper", "--job-dir", str(job_dir), "--linger-s", str(linger_s),
               "--", *entry]
    J.write_model(job_dir / J.JOB, J.Job(id=name, component="dataset", command=wrapped,
                                         params={"args": list(map(str, argv))}))
    r = subprocess.run([*opts, "-d", "--name", name, *container(wrapped)], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise LaunchError(f"docker run failed ({r.returncode}): {(r.stderr or r.stdout).strip()}")
    return Job(name, job_dir, time.time())


def wait(job: Job, timeout=7200.0) -> dict:
    """The job's exit code (`docker wait`; the container is not started with --rm, so the code cannot be lost to an
    early removal), then the container removed."""
    try:
        out = subprocess.run(["docker", "wait", job.container], capture_output=True, text=True,
                             timeout=timeout).stdout.strip()
        code = int(out) if out.lstrip("-").isdigit() else -1
    except subprocess.TimeoutExpired:
        code = -1
    subprocess.run(["docker", "rm", "-f", job.container], capture_output=True)
    f = job.dir / "result.json"
    return dict(exit=code, wall_s=round(time.time() - job.t0, 1),
                result=json.loads(f.read_text()) if f.is_file() else None, dir=str(job.dir))


def convert(manifests, out, turn, split=None, job_dir=None, **kw) -> dict:
    argv = ["convert", *[x for m in manifests for x in ("--manifest", str(m))], *(["--split", split] if split else []),
            "--turn", str(turn), "--out", str(out)]
    return wait(start(argv, job_dir, **kw))


def merge(datasets, out, job_dir=None, **kw) -> dict:
    return wait(start(["merge", "--out", str(out), *map(str, datasets)], job_dir, **kw))
