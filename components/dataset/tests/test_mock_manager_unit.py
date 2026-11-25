"""components/dataset/mock_manager.py: main() starts a job through launch.start / launch.wait, host only."""
import importlib.util
import json
import sys
import uuid

import pytest

from conftest import COMP


@pytest.fixture
def mm(monkeypatch):
    """The script loaded under a unique name, with launch.start / launch.wait recorded and faked."""
    name = f"mock_manager_probe_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, COMP / "mock_manager.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []
    handle = object()

    def start(job, job_dir):
        calls.append(("start", job, job_dir))
        return handle

    def wait(h):
        calls.append(("wait", h))
        return {"exit": 3, "result": {"dataset": "D", "episodes": 2, "frames": 9, "problems": ["x"]}}
    monkeypatch.setattr(mod.launch, "start", start)
    monkeypatch.setattr(mod.launch, "wait", wait)
    mod.calls, mod.handle = calls, handle
    return mod


# loading the script registers launch.py as dataset_launch and keeps it as the launch attribute
def test_loads_launch_module(mm):
    assert sys.modules["dataset_launch"] is mm.launch
    assert hasattr(mm.launch, "start") and hasattr(mm.launch, "LaunchError")


# main with --job-dir passes the job argv through unchanged, waits on the handle and returns its exit code
def test_job_dir_passthrough(mm, capsys, tmp_path):
    job = ["convert", "--manifest", "/runs/m.json", "--turn", "0", "--out=D"]
    rc = mm.main(["--job-dir", str(tmp_path), *job])
    assert rc == 3
    assert mm.calls == [("start", job, str(tmp_path)), ("wait", mm.handle)]
    assert json.loads(capsys.readouterr().out) == {"dataset": "D", "episodes": 2, "frames": 9, "problems": ["x"]}


# main without --job-dir starts the job with no job dir
def test_no_job_dir(mm, capsys):
    job = ["merge", "--dataset", "a", "--dataset", "b", "--out", "c"]
    assert mm.main(job) == 3
    assert mm.calls[0] == ("start", job, None)
    assert mm.calls[1] == ("wait", mm.handle)


# a LaunchError from start is an argparse usage error: exit 2 with the message on stderr
def test_launch_error_exits_2(mm, monkeypatch, capsys):
    def start(job, job_dir):
        raise mm.launch.LaunchError("the out dir is foreign")
    monkeypatch.setattr(mm.launch, "start", start)
    with pytest.raises(SystemExit) as e:
        mm.main(["convert", "--manifest", "m", "--turn", "0", "--out", "o"])
    assert e.value.code == 2
    assert "the out dir is foreign" in capsys.readouterr().err
