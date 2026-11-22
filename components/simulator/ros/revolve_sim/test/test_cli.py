"""`./revolve sim` and `./revolve validate` on the host, no sim: a fake `docker` and `nvidia-smi` first on PATH log
every call. The spec rules themselves are tested in common/tests/test_specs.py; here one case per refusal shows that
the CLI applies them. A `revolve sim` that passes its checks needs the scene's assets (and the simulator builds to
reach docker): those cases are skipped without them."""
import os
import subprocess

import pytest
from host_cases import REPO, SCENE_SPEC, pose, write_instance, snapshot

BASE = ["--scene", "kitchen_partnet", "--robot", "mobile_panda"]
ASSETS = (REPO / "assets" / "generated_partnet").is_dir()
BUILT = all((REPO / "ThirdParty" / d / "install" / "setup.bash").is_file() for d in ("gz_kura", "ros_gz_ws"))
needs_assets = pytest.mark.skipif(not ASSETS, reason="no assets/ (./revolve assets)")
needs_assets_and_builds = pytest.mark.skipif(not (ASSETS and BUILT), reason="no assets/ or simulator builds")


@pytest.fixture
def revolve(tmp_path):
    """Run ./revolve with a fake docker / nvidia-smi on PATH -> (CompletedProcess, the fakes' logged calls)."""
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    log = tmp_path / "fake_calls.log"
    for name in ("docker", "nvidia-smi"):
        (fake_bin / name).write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\nexit 0\n')
        (fake_bin / name).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "REVOLVE_GPU_PREFLIGHT")}
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"

    def run(*args, cwd=REPO):
        r = subprocess.run([str(REPO / "revolve"), *map(str, args)], capture_output=True, text=True, check=False,
                           cwd=cwd, env=env, timeout=120)
        return r, (log.read_text().splitlines() if log.exists() else [])

    return run


def ok_lines(r):
    return [ln for ln in r.stdout.splitlines() if ln.startswith("OK ")]


# ------------------------------------------------------------------ revolve sim
@pytest.mark.parametrize("extra", [["--record", "images"], ["--no-autostart"], ["--out", "{tmp}/out"]],
                         ids=["record", "no-autostart", "out"])
def test_sim_task_only_flags_need_task(revolve, tmp_path, extra):
    extra = [a.replace("{tmp}", str(tmp_path)) for a in extra]
    before = snapshot(tmp_path)
    r, calls = revolve("sim", *BASE, *extra)
    assert r.returncode == 1 and r.stderr.startswith(f"revolve sim: {extra[0]} needs --task"), r.stderr
    assert calls == [] and snapshot(tmp_path) == before        # nothing ran, --out not created


def duplicate_name(d):
    path = d / "scene.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(SCENE_SPEC.read_text() + "name: kitchen_partnet\n")
    return path


INSTANCE = "inst/scene.yaml"
BAD_SCENES = {   # id: (writes the bad scene file under tmp -> its path, what the refusal says)
    "unknown-pose": (lambda t: write_instance(t / INSTANCE, arm="no_such_pose"), "robot_start.arm 'no_such_pose'"),
    "arm-length": (lambda t: write_instance(t / INSTANCE, arm=pose("mobile_panda", "ready")[:5]),
                   "robot_start.arm has 5 joints, robot mobile_panda 7"),
    "nonzero-joint": (lambda t: write_instance(t / INSTANCE, joints={"top_door_hinge": 0.5}),
                      "nonzero initial joint positions"),
    "file-name": (lambda t: write_instance(t / "inst" / "seed_7.yaml"), "a scene instance file is named scene.yaml"),
    "duplicate-key": (lambda t: duplicate_name(t / "inst"), "duplicate key 'name'"),
}


@pytest.mark.parametrize("case, robot_args", [
    pytest.param("unknown-pose", ["--robot", "mobile_panda"], marks=needs_assets),
    pytest.param("arm-length", ["--robot", "mobile_panda", "--task", "open_fridge_door", "--no-autostart"],
                 marks=needs_assets),
    pytest.param("nonzero-joint", ["--robot", "mobile_panda"], marks=needs_assets),
    ("file-name", ["--robot", "mobile_panda"]),
])
def test_sim_refuses_a_bad_scene_before_docker(revolve, tmp_path, case, robot_args):
    """What the launch would refuse is refused on the host: exit 1, the scene file named, no docker, no GPU check.
    Without assets/ a missing model is the refusal (reported first), so the robot start cases need them."""
    make, needle = BAD_SCENES[case]
    path = make(tmp_path)
    before = snapshot(tmp_path)
    r, calls = revolve("sim", "--scene", path, *robot_args)
    assert r.returncode == 1 and r.stderr.startswith("revolve sim: "), r.stderr
    assert str(path) in r.stderr and needle in r.stderr, r.stderr
    assert calls == [] and snapshot(tmp_path) == before


@needs_assets_and_builds
def test_sim_starts_the_launch_with_its_args(revolve, tmp_path):
    out = tmp_path / "out"
    r, calls = revolve("sim", *BASE, "--task", "open_fridge_door", "--no-autostart", "--record", "no-images",
                       "--out", out)
    assert r.returncode == 0, r.stderr
    run = [c for c in calls if c.startswith("docker run ")]
    assert len(run) == 1, calls
    argv = run[0].split()
    for arg in ("scene:=/repo/specs/scenes/kitchen_partnet/scene.yaml", "robot:=mobile_panda", "task:=open_fridge_door",
                "record:=no-images", "autostart:=false", f"run_dir:={out}", f"{out}:{out}"):
        assert arg in argv, run[0]
    inst = write_instance(tmp_path / "x" / "scene.yaml", arm="ready")
    r, calls = revolve("sim", "--scene", inst, "--robot", "mobile_panda")
    assert r.returncode == 0, r.stderr
    run = [c for c in calls if c.startswith("docker run ")][-1]
    argv = run.split()
    assert f"scene:={inst}" in argv and f"{inst}:{inst}:ro" in argv and "task:=" not in run, run


@needs_assets
@pytest.mark.parametrize("output, code", [("| 0  NVIDIA RTX  ERR! |", 0), ("", 3)], ids=["err", "exit3"])
def test_sim_gpu_preflight_failure_refused(revolve, tmp_path, output, code):
    log = tmp_path / "fake_calls.log"
    (tmp_path / "fake_bin" / "nvidia-smi").write_text(
        f"#!/bin/sh\necho \"nvidia-smi $*\" >> \"{log}\"\necho '{output}'\nexit {code}\n")
    r, calls = revolve("sim", *BASE)
    assert r.returncode == 1 and "GPU preflight failed" in r.stderr and "Traceback" not in r.stderr, r.stderr
    assert any(c.startswith("nvidia-smi") for c in calls) and not any(c.startswith("docker") for c in calls), calls


# ------------------------------------------------------------------ revolve validate
def test_validate_checks_robot_start_with_a_robot(revolve, tmp_path):
    good = write_instance(tmp_path / "good" / "scene.yaml", arm="ready")
    bad = write_instance(tmp_path / "bad" / "scene.yaml", arm="no_such_pose")
    r, _ = revolve("validate", good, "--robot", "mobile_so101", "--no-assets")
    assert r.returncode == 0 and len(ok_lines(r)) == 1 and "mobile_so101" in r.stdout, r.stderr
    r, _ = revolve("validate", bad, "--robot", "mobile_so101", "--no-assets")
    assert r.returncode == 1 and str(bad) in r.stderr and "robot_start.arm 'no_such_pose'" in r.stderr, r.stderr
    r, _ = revolve("validate", bad, "--no-assets")       # without --robot the arm is checked against no robot
    assert r.returncode == 0 and len(ok_lines(r)) == 1, r.stderr


@pytest.mark.parametrize("case", ["nonzero-joint", "file-name", "duplicate-key"])
def test_validate_refuses_a_bad_file(revolve, tmp_path, case):
    make, needle = BAD_SCENES[case]
    path = make(tmp_path)
    good = write_instance(tmp_path / "good" / "scene.yaml")
    r, calls = revolve("validate", path, good, "--no-assets")
    assert r.returncode == 1 and needle in r.stderr, r.stderr
    assert len(ok_lines(r)) == 1 and calls == []        # the good file still gets its OK line


def test_validate_from_another_directory(revolve, tmp_path):
    """Every repo spec, and a relative path resolved against the caller's directory, not the repo."""
    from_repo, _ = revolve("validate", "--no-assets")
    outside, _ = revolve("validate", "--no-assets", cwd=tmp_path)
    assert outside.returncode == 0 and len(ok_lines(outside)) == len(ok_lines(from_repo)) > 0, outside.stderr
    write_instance(tmp_path / "inst" / "scene.yaml")
    r, _ = revolve("validate", "inst/scene.yaml", "--no-assets", cwd=tmp_path)
    assert r.returncode == 0 and len(ok_lines(r)) == 1, r.stderr
