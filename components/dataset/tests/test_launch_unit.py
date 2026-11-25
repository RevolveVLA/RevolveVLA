"""components/dataset/launch.py on the host, without docker (DATASET-v2.md sections 4, 7, 12 "Hub never pushed"): the
module's constants, docker_cmd's mounts / network / env, its LaunchErrors (a bad argv parsed with the entrypoint's
own parser, a missing or foreign Rosetta build), and start / wait against a fake `docker`. Every other subprocess call
is blocked: these tests must never start docker."""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from conftest import REPO, no_env_image
from dataset import cli


@pytest.fixture(autouse=True)
def no_subprocess(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError(f"unit tests must not run commands (docker): {a[:1]}")
    for fn in ("run", "call", "check_call", "check_output", "Popen"):
        monkeypatch.setattr(subprocess, fn, blocked)


GOOD_STAMP = "deadbeef lyrical patches:none\n"      # build.sh: <rosetta sha> <ROS distro> patches:<digest>


@pytest.fixture
def L(launch, tmp_path, monkeypatch):
    """The launch module with INSTALL = a tmp dir holding a good BUILD_OK."""
    inst = tmp_path / "install"
    inst.mkdir(exist_ok=True)
    (inst / "BUILD_OK").write_text(GOOD_STAMP)
    monkeypatch.setattr(launch, "INSTALL", inst)
    return launch


def real(p):
    return os.path.realpath(str(p))


def mounts(cmd, resolve=True):
    """[(source, target, read_only)] from -v / --volume / --mount arguments; resolve=False keeps the paths as written
    (only normalised, symlinks not followed)."""
    out, i = [], 0
    while i < len(cmd):
        a = cmd[i]
        spec = kind = None
        if a in ("-v", "--volume", "--mount"):
            kind, spec, i = a, cmd[i + 1], i + 2
        elif a.startswith("--volume=") or a.startswith("--mount="):
            kind, spec, i = a.split("=", 1)[0], a.split("=", 1)[1], i + 1
        elif a.startswith("-v") and len(a) > 2:
            kind, spec, i = "-v", a[2:], i + 1
        else:
            i += 1
            continue
        if kind == "--mount":
            kv = dict((p.split("=", 1) + ["true"])[:2] for p in spec.split(","))
            src = kv.get("source", kv.get("src"))
            dst = kv.get("target", kv.get("destination", kv.get("dst")))
            ro = kv.get("readonly", kv.get("ro", "false")).lower() in ("true", "1")
        else:
            parts = spec.split(":")
            src, dst = parts[0], parts[1]
            ro = len(parts) > 2 and "ro" in parts[2].split(",")
        norm = real if resolve else os.path.normpath
        out.append((norm(src), norm(dst), ro))
    return out


def full_cmd(L, argv, **kw):
    """docker_cmd's options and entrypoint as the one `docker run` command run() builds."""
    opts, entry = L.docker_cmd(argv, **kw)
    return [*opts, "--rm", *L.container(entry)]


def mount_at(cmd, target):
    ms = [m for m in mounts(cmd) if m[1] == real(target)]
    assert ms, f"nothing mounted at {target}: {mounts(cmd)}"
    assert len({(m[0], m[2]) for m in ms}) == 1, f"conflicting mounts at {target}: {ms}"
    return ms[0]


def env(cmd):
    out, i = {}, 0
    while i < len(cmd):
        a = cmd[i]
        if a in ("-e", "--env"):
            k, _, v = cmd[i + 1].partition("=")
            out[k], i = v, i + 2
        elif a.startswith("--env="):
            k, _, v = a[len("--env="):].partition("=")
            out[k], i = v, i + 1
        else:
            i += 1
    return out


def network(cmd):
    for i, a in enumerate(cmd):
        if a in ("--network", "--net"):
            return cmd[i + 1]
        if a.startswith("--network=") or a.startswith("--net="):
            return a.split("=", 1)[1]
    return None


def convert_argv(tmp_path, out=None, manifest_dir=None):
    mdir = manifest_dir or tmp_path / "producer"
    mdir.mkdir(parents=True, exist_ok=True)
    m = mdir / "manifest.json"
    m.write_text('{"task": "open_fridge_door", "robot": "mobile_panda", "episodes": []}')
    out = out or tmp_path / "outs" / "ds"
    out.parent.mkdir(parents=True, exist_ok=True)
    return ["convert", "--manifest", str(m), "--turn", "0", "--out", str(out)], m, out


# ------------------------------------------------------------------ module constants

def test_constants(launch):
    # launch: REPO, IMAGE (env REVOLVE_DATASET_IMAGE, default revolve-dataset:latest), INSTALL, LaunchError
    assert real(launch.REPO) == real(REPO)
    assert real(launch.INSTALL) == real(REPO / "ThirdParty" / "rosetta" / "install")
    if no_env_image():
        assert launch.IMAGE == "revolve-dataset:latest"
    else:
        assert launch.IMAGE == os.environ["REVOLVE_DATASET_IMAGE"]
    assert issubclass(launch.LaunchError, RuntimeError)


# ------------------------------------------------------------------ docker_cmd: mounts, network, env

def test_convert_mounts(L, tmp_path):
    # launch.docker_cmd: repo ro at /repo and at its host path, runs/ rw at both, install ro at
    # /opt/rosetta/install, an outside-repo manifest dir ro at its own path, the out's parent rw
    argv, m, out = convert_argv(tmp_path)
    cmd = full_cmd(L, argv)
    assert isinstance(cmd, list) and all(isinstance(a, str) for a in cmd)
    assert cmd[0].endswith("docker")
    assert L.IMAGE in cmd
    assert mount_at(cmd, "/repo") == (real(REPO), "/repo", True)
    assert mount_at(cmd, real(REPO)) == (real(REPO), real(REPO), True)
    assert mount_at(cmd, "/repo/runs") == (real(REPO / "runs"), "/repo/runs", False)
    assert mount_at(cmd, real(REPO / "runs")) == (real(REPO / "runs"), real(REPO / "runs"), False)
    assert mount_at(cmd, "/opt/rosetta/install") == (real(L.INSTALL), "/opt/rosetta/install", True)
    assert mount_at(cmd, real(m.parent)) == (real(m.parent), real(m.parent), True)
    assert mount_at(cmd, real(out.parent)) == (real(out.parent), real(out.parent), False)


def test_offline_and_no_network(L, tmp_path):
    # launch.docker_cmd: --network none without job_dir; DATASET-v2 4/12: HF_HUB_OFFLINE=1, no token
    argv, _, _ = convert_argv(tmp_path)
    cmd = full_cmd(L, argv)
    assert network(cmd) == "none"
    e = env(cmd)
    assert e.get("HF_HUB_OFFLINE") == "1"
    assert not any(k in e for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")), e


def test_job_dir_needs_a_network(L, tmp_path):
    # launch.docker_cmd: --network none only without job_dir (the wrapper's HTTP on port 8000 must be reachable at the
    # container's address); still offline for the Hub; the job dir writable at its own path
    argv, _, _ = convert_argv(tmp_path)
    jd = tmp_path / "jobs" / "j1"
    jd.mkdir(parents=True)
    cmd = full_cmd(L, argv, job_dir=jd)
    assert network(cmd) != "none"
    assert env(cmd).get("HF_HUB_OFFLINE") == "1"
    assert mount_at(cmd, real(jd)) == (real(jd), real(jd), False)


def test_several_manifests_each_mounted(L, tmp_path):
    # launch.docker_cmd: EACH manifest's directory (outside the repo) ro at its own path
    argv, m1, out = convert_argv(tmp_path, manifest_dir=tmp_path / "p1")
    m2 = tmp_path / "p2" / "manifest.json"
    m2.parent.mkdir()
    m2.write_text("{}")
    argv = argv[:3] + ["--manifest", str(m2)] + argv[3:]
    cmd = full_cmd(L, argv)
    assert mount_at(cmd, real(m1.parent))[2] is True
    assert mount_at(cmd, real(m2.parent))[2] is True


def test_manifest_and_out_inside_runs(L):
    # launch.docker_cmd: --out under REPO/runs/ is fine (no LaunchError), runs/ writable
    d = REPO / "runs" / f"dataset_unit_never_created_{uuid.uuid4().hex[:8]}"      # unique: never a stale dir
    m, out = d / "manifest.json", d / "ds"
    cmd = full_cmd(L, ["convert", "--manifest", str(m), "--turn", "0", "--out", str(out)])
    assert mount_at(cmd, real(REPO / "runs"))[2] is False
    assert not out.exists() and not out.parent.exists(), "docker_cmd must not build anything"


def test_merge_mounts(L, tmp_path):
    # launch.docker_cmd: each input dataset (outside the repo) ro at its own path, the out's parent rw
    a, b = tmp_path / "ds_a", tmp_path / "elsewhere" / "ds_b"
    for d in (a, b):
        (d / "meta").mkdir(parents=True)
    out = tmp_path / "merged" / "ds"
    out.parent.mkdir()
    cmd = full_cmd(L, ["merge", "--out", str(out), str(a), str(b)])
    assert mount_at(cmd, real(a)) == (real(a), real(a), True)
    assert mount_at(cmd, real(b)) == (real(b), real(b), True)
    assert mount_at(cmd, real(out.parent)) == (real(out.parent), real(out.parent), False)
    assert mount_at(cmd, "/repo")[2] is True


def test_relative_paths_against_cwd(L, tmp_path):
    # launch.docker_cmd(cwd=...): relative manifest / out paths are the caller's, mounted at their real path
    argv, m, out = convert_argv(tmp_path)
    rel = ["convert", "--manifest", os.path.relpath(m, tmp_path), "--turn", "0",
           "--out", os.path.relpath(out, tmp_path)]
    cmd = full_cmd(L, rel, cwd=tmp_path)
    assert mount_at(cmd, real(m.parent))[2] is True
    assert mount_at(cmd, real(out.parent))[2] is False


def test_symlinked_paths_mounted_as_written(L, tmp_path):
    # a manifest dir and an out parent written through symlinks are mounted at the written path too
    _, m, _ = convert_argv(tmp_path, manifest_dir=tmp_path / "real_prod")
    (tmp_path / "real_outs").mkdir()
    mlink, olink = tmp_path / "prod_link", tmp_path / "outs_link"
    mlink.symlink_to(tmp_path / "real_prod", target_is_directory=True)
    olink.symlink_to(tmp_path / "real_outs", target_is_directory=True)
    written_m = str(mlink / "manifest.json")
    opts, entry = L.docker_cmd(["convert", "--manifest", written_m, "--turn", "0", "--out", str(olink / "ds")])
    raw = mounts(opts, resolve=False)
    at_out = [(real(s), ro) for s, d, ro in raw if d == os.path.abspath(olink)]
    assert set(at_out) == {(real(tmp_path / "real_outs"), False)}, raw
    at_manifest = [(real(s), ro) for s, d, ro in raw if d == os.path.abspath(mlink)]
    assert set(at_manifest) == {(real(m.parent), True)}, raw
    got = cli.parser().parse_args(entry[3:]).manifest
    assert written_m in (got if isinstance(got, list) else [got])


def test_option_value_starting_with_a_dash_survives(L, tmp_path):
    # a value starting with '-' is re-emitted so the entrypoint's parser reads it back unchanged
    argv, _, _ = convert_argv(tmp_path)
    _, entry = L.docker_cmd([*argv[:-2], "--split=-held", *argv[-2:]])
    assert entry[1:3] == ["-m", "dataset.cli"]
    assert cli.parser().parse_args(entry[3:]).split == "-held"


# ------------------------------------------------------------------ docker_cmd: LaunchError

@pytest.mark.parametrize("sub", ["components/dataset/ds", "specs/ds", "ds", "runs/../specs/ds"])
def test_out_inside_repo_outside_runs(L, tmp_path, sub):
    # launch.docker_cmd: LaunchError when --out is inside the repo but not under runs/
    argv, _, _ = convert_argv(tmp_path, out=tmp_path / "unused" / "ds")
    argv[-1] = str(REPO / sub)
    with pytest.raises(L.LaunchError):
        L.docker_cmd(argv)
    assert not (REPO / sub).exists()


def test_out_inside_repo_relative_to_cwd(L, tmp_path):
    # launch.docker_cmd: the same refusal for a relative --out resolved against cwd
    argv, _, _ = convert_argv(tmp_path)
    argv[-1] = "components/dataset/ds_relative"
    with pytest.raises(L.LaunchError):
        L.docker_cmd(argv, cwd=REPO)


def test_merge_out_inside_repo_outside_runs(L, tmp_path):
    # launch.docker_cmd: the --out rule holds for merge too
    a = tmp_path / "a"
    (a / "meta").mkdir(parents=True)
    with pytest.raises(L.LaunchError):
        L.docker_cmd(["merge", "--out", str(REPO / "specs" / "merged"), str(a)])


def test_missing_build_ok(launch, tmp_path, monkeypatch):
    # launch.docker_cmd: LaunchError when INSTALL/BUILD_OK is missing (Rosetta not built)
    inst = tmp_path / "install"
    inst.mkdir()
    monkeypatch.setattr(launch, "INSTALL", inst)
    argv, _, _ = convert_argv(tmp_path)
    with pytest.raises(launch.LaunchError):
        launch.docker_cmd(argv)


def test_missing_install_dir(launch, tmp_path, monkeypatch):
    # launch.docker_cmd: LaunchError when the install does not exist at all
    monkeypatch.setattr(launch, "INSTALL", tmp_path / "never_built")
    argv, _, _ = convert_argv(tmp_path)
    with pytest.raises(launch.LaunchError):
        launch.docker_cmd(argv)


# ------------------------------------------------------------------ run(): LaunchError -> exit 2

def test_run_launch_error_exits_2(launch, tmp_path, monkeypatch, capsys):
    # launch.run: LaunchError -> exit 2 with a message (and no docker started: subprocess is blocked)
    monkeypatch.setattr(launch, "INSTALL", tmp_path / "never_built")
    argv, _, out = convert_argv(tmp_path)
    assert launch.run(argv) == 2
    cap = capsys.readouterr()
    assert (cap.out + cap.err).strip(), "a message"
    assert not out.exists()


def test_run_out_in_repo_exits_2(L, tmp_path, capsys):
    # launch.run: the --out refusal is a LaunchError -> exit 2
    argv, _, _ = convert_argv(tmp_path)
    argv[-1] = str(REPO / "specs" / "ds_never")
    assert L.run(argv) == 2
    assert not (REPO / "specs" / "ds_never").exists()


# ------------------------------------------------------------------ the entrypoint's parser decides the mounts

@pytest.mark.parametrize("cmd", ["convert", "merge"])
def test_equals_forms_mount_the_same(L, tmp_path, cmd):
    # `--out=D` and `--manifest=M` get the mounts of the two-token forms (the out's parent rw, the manifest's dir ro)
    argv, m, out = convert_argv(tmp_path)
    if cmd == "merge":
        a = tmp_path / "ds_a"
        (a / "meta").mkdir(parents=True)
        two, eq = ["merge", "--out", str(out), str(a)], ["merge", f"--out={out}", str(a)]
    else:
        two, eq = argv, ["convert", f"--manifest={m}", "--turn", "0", f"--out={out}"]
    got = mounts(L.docker_cmd(eq)[0])
    assert sorted(got) == sorted(mounts(L.docker_cmd(two)[0]))
    assert (real(out.parent), real(out.parent), False) in got
    if cmd == "convert":
        assert (real(m.parent), real(m.parent), True) in got


def test_docker_cmd_refuses_an_abbreviated_option(L, tmp_path):
    argv, _, _ = convert_argv(tmp_path)
    with pytest.raises(L.LaunchError):
        L.docker_cmd([*argv[:-2], "--ou", argv[-1]])


def test_run_refuses_an_abbreviated_option(L, tmp_path, capsys):
    # exit 2 with a message, no docker started
    argv, _, out = convert_argv(tmp_path)
    assert L.run([*argv[:-2], "--ou", argv[-1]]) == 2
    assert capsys.readouterr().err.strip()
    assert not out.exists()


# ------------------------------------------------------------------ BUILD_OK names the image's ROS distro

@pytest.mark.parametrize("stamp", ["deadbeef\n", "deadbeef jazzy patches:abc\n", "deadbeef lyrical-old patches:abc\n"],
                         ids=["no-distro", "jazzy", "lyrical-prefix"])
def test_build_ok_must_name_lyrical(L, tmp_path, stamp):
    # Rosetta's install was built for one ROS distro: an install whose stamp's second field is not exactly `lyrical`
    # (an old Jazzy build) is refused up front instead of failing on an obscure import in the container
    argv, _, _ = convert_argv(tmp_path)
    (L.INSTALL / "BUILD_OK").write_text("deadbeef lyrical\n")        # no patches field: the distro is what counts
    L.docker_cmd(argv)
    (L.INSTALL / "BUILD_OK").write_text(stamp)
    with pytest.raises(L.LaunchError):
        L.docker_cmd(argv)


# ------------------------------------------------------------------ start / wait against a fake docker

class FakeDocker:
    """subprocess.run stand-in: records every command and answers `docker run` (a container id, or exit 125 with a
    recognisable stderr when run_fails) and `docker wait` (wait_code)."""

    def __init__(self, run_fails=False, wait_code=3):
        self.calls, self.run_fails, self.wait_code = [], run_fails, wait_code

    @staticmethod
    def sub(cmd):
        """The docker subcommand of cmd ('run', 'wait', 'rm', ...) or None."""
        if not cmd or not cmd[0].endswith("docker"):
            return None
        return next((a for a in cmd[1:] if not a.startswith("-")), None)

    def __call__(self, cmd, *a, **k):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        rc, out, err = 0, "", ""
        if self.sub(cmd) == "run":
            if self.run_fails:
                rc, err = 125, "docker: Error response from daemon: conflict-marker-7f3a: name already in use."
            else:
                out = "c0ffee1234567890\n"
        elif self.sub(cmd) == "wait":
            out = f"{self.wait_code}\n"
        return subprocess.CompletedProcess(cmd, rc, out, err)

    def of(self, sub):
        return [c for c in self.calls if self.sub(c) == sub]


@pytest.fixture
def fake_docker(L, monkeypatch):
    def make(**kw):
        fd = FakeDocker(**kw)
        monkeypatch.setattr(subprocess, "run", fd)
        return fd
    return make


def test_start_failure_raises_launch_error_with_stderr(L, fake_docker, tmp_path):
    fd = fake_docker(run_fails=True)
    argv, _, _ = convert_argv(tmp_path)
    with pytest.raises(L.LaunchError) as e:
        L.start(argv, tmp_path / "jobs" / "j1")
    assert "conflict-marker-7f3a" in str(e.value)
    assert fd.of("run"), "docker run was never called"


def test_start_detached_without_rm(L, fake_docker, tmp_path):
    # the job container is not started with --rm: its exit code must survive until `docker wait`
    fd = fake_docker()
    argv, _, _ = convert_argv(tmp_path)
    job = L.start(argv, tmp_path / "jobs" / "j2")
    (run,) = fd.of("run")
    assert "--rm" not in run and not any(a.startswith("--rm=") for a in run), run
    assert "-d" in run and run[run.index("--name") + 1] == job.container


def test_start_default_job_dir_from_out(L, fake_docker, tmp_path):
    # without a job dir: <out's parent>/jobs/dataset_<kind>_<out's name>, writable in the container, job.json in it
    # naming the wrapped entrypoint
    fd = fake_docker()
    argv, _, out = convert_argv(tmp_path)
    job = L.start(argv)
    jd = out.parent / "jobs" / f"dataset_convert_{out.name}"
    assert real(job.dir) == real(jd) == real(L.default_job_dir(out, "convert"))
    (run,) = fd.of("run")
    assert any(not ro and src == dst and Path(real(jd)).is_relative_to(src) for src, dst, ro in mounts(run)), run
    doc = json.loads((jd / "job.json").read_text())
    assert doc["component"] == "dataset" and doc["id"] == job.container
    assert "revolve_common.wrapper" in doc["command"] and f"--out={out}" in doc["command"]   # --opt=value form


def test_start_job_dir_inside_the_repo_only_under_runs(L, fake_docker, tmp_path):
    argv, _, _ = convert_argv(tmp_path)
    fd = fake_docker()
    with pytest.raises(L.LaunchError):
        L.start(argv, REPO / "specs" / "jobs_never")
    assert not (REPO / "specs" / "jobs_never").exists() and not fd.calls


def test_wait_reads_the_exit_code_then_removes(L, fake_docker, tmp_path):
    fd = fake_docker(wait_code=3)
    argv, _, _ = convert_argv(tmp_path)
    jd = tmp_path / "jobs" / "j3"
    job = L.start(argv, jd)
    (jd / "result.json").write_text(json.dumps({"state": "failed", "returncode": 3,
                                                "progress": {"stage": "failed", "problems": ["frames: x"]}}))
    n0 = len(fd.calls)
    r = L.wait(job, 60)
    after = fd.calls[n0:]
    waits = [i for i, c in enumerate(after) if FakeDocker.sub(c) == "wait" and job.container in c]
    rms = [i for i, c in enumerate(after) if FakeDocker.sub(c) == "rm" and job.container in c]
    assert waits and rms and min(waits) < max(rms), f"docker rm must come after docker wait: {after}"
    assert r["exit"] == 3 and r["result"]["state"] == "failed" and r["dir"] == str(job.dir)


_REAL_POPEN = subprocess.Popen      # kept before the autouse fixture blocks subprocess


# importing launch.py leaves sys.path exactly as it was
def test_import_leaves_sys_path_unchanged():
    path = str(REPO / "components" / "dataset" / "launch.py")
    probe = (
        "import importlib.util, json, sys\n"
        "before = list(sys.path)\n"
        f"spec = importlib.util.spec_from_file_location('dataset_launch_probe', {path!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['dataset_launch_probe'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "print(json.dumps({'before': before, 'after': list(sys.path)}))\n")
    env = {**os.environ, "PYTHONPATH": str(REPO / "common")}
    with _REAL_POPEN([sys.executable, "-c", probe], cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True) as proc:
        out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    doc = json.loads(out.strip().splitlines()[-1])
    assert doc["after"] == doc["before"]
