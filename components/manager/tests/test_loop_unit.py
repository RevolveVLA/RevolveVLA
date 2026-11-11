"""Unit tests of loop.py's launch paths with a fake docker client (no containers): a step container docker cannot
start is removed and counts as a failed attempt; the GPU preflight (mocked) stops the experiment before a GPU step
and while one runs, leaving the step resumable."""
import sys
import types
from pathlib import Path

import docker
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "components" / "manager" / "src")]
from manager import loop as L  # noqa: E402
from manager import plan as P  # noqa: E402

from revolve_common import specs  # noqa: E402

SPEC = REPO / "specs" / "experiments" / "fridge_mini.yaml"


class FakeContainer:
    def __init__(self, name, labels, status="created"):
        self.name, self.labels, self.removed, self.status = name, labels, False, status
        self.attrs = {"NetworkSettings": {"Networks": {}}}

    def remove(self, force=False):
        self.removed = True

    def stop(self, timeout=None):
        self.status = "exited"

    def logs(self):
        return b""


class FakeContainers:
    def __init__(self, start_error=None):
        self.start_error, self.made = start_error, []

    def run(self, image, command=None, **kw):
        c = FakeContainer(kw.get("name"), kw["labels"], status="running")
        self.made.append(c)
        if self.start_error:
            raise docker.errors.APIError("500 Server Error", explanation=self.start_error)
        return c

    def get(self, name):
        for c in self.made:
            if c.name == name and not c.removed:
                return c
        raise docker.errors.NotFound(name)

    def list(self, all=False, filters=None):
        k, v = filters["label"].split("=")
        return [c for c in self.made if c.labels.get(k) == v and not c.removed]


@pytest.fixture
def manager(tmp_path, monkeypatch):
    def make(dk):
        m = object.__new__(L.Manager)
        m.spec, m.r = SPEC, specs.load_experiment(SPEC)
        m.e, m.dk, m.poll_s = m.r.experiment, dk, 0.01
        m.paths = P.Paths(tmp_path / "runs" / "fridge_mini")
        m.state = P.new_state("fridge_mini", SPEC)
        return m
    monkeypatch.setattr(L.P, "step_params", lambda *a: {"watch": str(tmp_path)})
    monkeypatch.setattr(L, "container_kw", lambda env=None: dict(detach=True, environment={}))
    monkeypatch.delenv("REVOLVE_STUB", raising=False)
    return make


def fake_dk(**kw):
    return types.SimpleNamespace(containers=FakeContainers(**kw), networks=types.SimpleNamespace(list=lambda **k: []))


def gpu_answers(monkeypatch, answers):
    """Mocked gpu.check_docker: pops True (healthy) / an error text per call."""
    calls = []

    def check(dk, image):
        calls.append(image)
        a = answers.pop(0)
        if a is not True:
            raise L.gpu.GpuError(f"GPU preflight failed: {a}")
        return "ok"
    monkeypatch.setattr(L.gpu, "check_docker", check)
    return calls


def test_step_container_docker_cannot_start(manager):
    dk = fake_dk(start_error="nvidia-container-cli: nvml error: gpu requires reset")
    m = manager(dk)
    m.run_step(0, "dataset")
    s = m.state["turns"][0]["steps"]["dataset"]
    assert [a["state"] for a in s["attempts"]] == ["failed", "failed"]
    assert "docker could not start" in s["attempts"][0]["error"] and "gpu requires reset" in s["attempts"][0]["error"]
    assert s["state"] == "failed" and "failed twice" in m.state["error"]
    assert len(dk.containers.made) == 2 and all(c.removed for c in dk.containers.made)


def test_gpu_preflight_before_a_gpu_step(manager, monkeypatch):
    dk = fake_dk()
    m = manager(dk)
    calls = gpu_answers(monkeypatch, ["nvidia-smi reports 'ERR!'"])
    m.state["turns"][0]["steps"]["generate"]["state"] = "running"
    monkeypatch.setattr(L.P, "next_action", lambda st, mt: ("run", 0, "generate"))
    assert m.run() == 1
    st = m.state
    assert st["status"] == "failed" and st["stop_reason"] == "gpu_unhealthy" and "ERR!" in st["error"]
    s = st["turns"][0]["steps"]["generate"]
    assert s["state"] == "pending" and s["attempts"] == [] and dk.containers.made == []   # resumable, nothing started
    assert calls == [L.IMAGE]


def test_no_preflight_for_cpu_steps_and_remote_training(manager, monkeypatch):
    m = manager(fake_dk())
    assert m.e.training.backend == "gpu_machine"
    assert not m.needs_gpu("dataset") and not m.needs_gpu("train")
    assert all(m.needs_gpu(x) for x in L.GPU_STEPS)
    monkeypatch.setenv("REVOLVE_STUB", "x.json")
    assert not m.needs_gpu("generate")


def test_gpu_dies_while_a_step_runs(manager, monkeypatch):
    dk = fake_dk()
    m = manager(dk)
    m.gpu_check_s = 0.0
    gpu_answers(monkeypatch, [True, "nvidia-smi reports 'requires reset'"])
    monkeypatch.setattr(m, "poll", lambda c, jd: {})
    with pytest.raises(L.GpuUnhealthy, match="requires reset"):
        m.run_step(0, "eval_heldout")
    a = m.state["turns"][0]["steps"]["eval_heldout"]["attempts"]
    assert len(a) == 1 and a[0]["state"] == "gpu_unhealthy" and "requires reset" in a[0]["error"]
    assert dk.containers.made[0].removed                      # the step was cancelled and removed


def test_resume_accept_spec_change(tmp_path, monkeypatch):
    """An edited yaml: the manager refuses it; resume --accept-spec-change records the new sha256 (old one in
    spec_changes) and keeps the done steps."""
    from manager import cli
    name = "mt_accept_spec"
    (tmp_path / "specs" / "experiments").mkdir(parents=True)
    for d in ("tasks", "robots", "scenes"):
        (tmp_path / "specs" / d).symlink_to(REPO / "specs" / d)
    spec = tmp_path / "specs" / "experiments" / f"{name}.yaml"
    spec.write_text(SPEC.read_text().replace("name: fridge_mini", f"name: {name}"))
    paths = P.Paths(REPO / "runs" / name)
    st = P.new_state(name, spec)
    st["turns"][0]["steps"]["generate"]["state"] = "done"
    paths.exp.mkdir(parents=True, exist_ok=True)
    try:
        P.save(st, paths)
        spec.write_text(spec.read_text().replace("steps: 2000", "steps: 3000"))
        monkeypatch.setattr(L.docker, "from_env", lambda: types.SimpleNamespace(containers=FakeContainers()))
        with pytest.raises(SystemExit, match="--accept-spec-change"):
            L.Manager(spec)
        runs = []
        fake = types.SimpleNamespace(containers=types.SimpleNamespace(
            get=lambda n: (_ for _ in ()).throw(docker.errors.NotFound(n)), run=lambda *a, **kw: runs.append(a)),
            images=types.SimpleNamespace(get=lambda i: None))
        monkeypatch.setattr(cli.docker, "from_env", lambda: fake)
        monkeypatch.setattr(L, "container_kw", lambda env=None: {})
        assert cli.start(str(spec), resume=True, accept_spec_change=True) == 0 and len(runs) == 1
        new = P.read(paths.exp / "state.json")
        assert new["spec_sha256"] == P.sha256(spec) and new["spec_changes"][0]["old"] == st["spec_sha256"]
        assert new["turns"][0]["steps"]["generate"]["state"] == "done"
        L.Manager(spec)                                  # accepted: no refusal now
    finally:
        import shutil
        shutil.rmtree(paths.exp, ignore_errors=True)
