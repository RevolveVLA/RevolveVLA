"""GPU preflight (revolve_common.gpu) against a mocked nvidia-smi and a fake docker client. BROKEN is the laptop's
real nvidia-smi table after Xid 119/154: exit 0, CSV queries normal, ERR! in the table."""
import os
import stat

import docker
import pytest

from revolve_common import gpu

HEALTHY = """+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.04             Driver Version: 580.159.04     CUDA Version: 13.0     |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   45C    P8              3W /   80W |      13MiB /   8151MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
"""
BROKEN = """+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.04             Driver Version: 580.159.04     CUDA Version: 13.0     |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 ...    On  |   00000000:01:00.0 N/A |                  N/A |
|ERR!  ERR! ERR!             N/A  /  N/A  |      13MiB /   8151MiB |     N/A      Default |
|                                         |                        |                 ERR! |
+-----------------------------------------+------------------------+----------------------+
"""


@pytest.fixture
def smi(tmp_path, monkeypatch):
    """smi(text, code=0, hang=False): a fake nvidia-smi first on PATH."""
    monkeypatch.delenv("REVOLVE_GPU_PREFLIGHT", raising=False)

    def make(text, code=0, hang=False):
        (tmp_path / "out.txt").write_text(text)
        exe = tmp_path / "nvidia-smi"
        exe.write_text(f"#!/bin/sh\n{'sleep 30' if hang else ''}\ncat {tmp_path / 'out.txt'}\nexit {code}\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    return make


def test_healthy(smi):
    smi(HEALTHY)
    assert gpu.check() == "ok"


def test_err_table_with_exit_0(smi):
    smi(BROKEN)
    with pytest.raises(gpu.GpuError, match="ERR!.*reset"):
        gpu.check()


def test_requires_reset_and_exit_code(smi):
    smi("    Product Brand    : GPU requires reset\n")
    with pytest.raises(gpu.GpuError, match="requires reset"):
        gpu.check()
    smi("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n", code=9)
    with pytest.raises(gpu.GpuError, match="exit 9: NVIDIA-SMI has failed"):
        gpu.check()


def test_hang_times_out(smi):
    smi(HEALTHY, hang=True)
    with pytest.raises(gpu.GpuError, match="did not answer within 1 s"):
        gpu.check(timeout_s=1)


def test_no_nvidia_smi_and_disabled(smi, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert gpu.check() == "not checked"
    smi(BROKEN)
    monkeypatch.setenv("REVOLVE_GPU_PREFLIGHT", "0")
    assert gpu.check() == "disabled"


def test_launch_check_caches_only_healthy(smi, monkeypatch):
    monkeypatch.setattr(gpu, "_last_ok", 0.0)
    smi(BROKEN)
    with pytest.raises(gpu.GpuError):
        gpu.launch_check()
    smi(HEALTHY)
    assert gpu.launch_check() == "ok"
    smi(BROKEN)
    assert gpu.launch_check() == "ok"              # healthy within CACHE_S: not re-run
    monkeypatch.setattr(gpu, "_last_ok", 0.0)
    with pytest.raises(gpu.GpuError):
        gpu.launch_check()


class Probe:
    def __init__(self, logs=b"", code=0, start_error=None):
        self._logs, self.code, self.start_error, self.removed, self.kw = logs, code, start_error, False, None

    def start(self):
        if self.start_error:
            raise docker.errors.APIError("500 Server Error", explanation=self.start_error)

    def wait(self, timeout=None):
        return {"StatusCode": self.code}

    def logs(self):
        return self._logs

    def remove(self, force=False):
        self.removed = True


class FakeDocker:
    def __init__(self, probe):
        self.probe = probe
        self.containers = self

    def create(self, image, command=None, **kw):
        self.probe.kw = dict(image=image, **kw)
        return self.probe


@pytest.mark.parametrize("probe, ok, match", [
    (Probe(HEALTHY.encode()), True, None),
    (Probe(BROKEN.encode()), False, "ERR!"),
    (Probe(start_error="nvidia-container-cli: initialization error: nvml error: gpu requires reset"), False,
     "docker could not start a GPU container: .*gpu requires reset"),
    (Probe(b"Failed to initialize NVML: Unknown Error\n", code=255), False, "exit 255"),
])
def test_check_docker(monkeypatch, probe, ok, match):
    monkeypatch.delenv("REVOLVE_GPU_PREFLIGHT", raising=False)
    dk = FakeDocker(probe)
    if ok:
        assert gpu.check_docker(dk, "img") == "ok"
    else:
        with pytest.raises(gpu.GpuError, match=match):
            gpu.check_docker(dk, "img")
    assert probe.removed and probe.kw["entrypoint"] == ["nvidia-smi"]
    assert probe.kw["environment"]["NVIDIA_DRIVER_CAPABILITIES"] == "utility"   # no compute: no workload
