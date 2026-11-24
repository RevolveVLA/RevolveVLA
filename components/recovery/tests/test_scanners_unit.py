"""plugins/scanners (local + container), scan_client and the rosbag2 decoding of bag_io.py (without rosbag2_py)."""
import json
import math
from types import SimpleNamespace as NS

import fake_recovery as F
import pytest
from recovery import bag_io, scan_client
from recovery.core import spec as S
from recovery.core.errors import RecoveryError
from recovery.plugins.scanners.container import ContainerScanner
from recovery.plugins.scanners.local import LocalScanner
from revolve_common.sim_session import ClientContainer


def spec(tmp_path, run):
    return S.parse({"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": str(run)}, "out": str(tmp_path / "out"),
                    "phases": {"generator": {"plugin": "fake_recovery"}, "quota": {"grasp": 1}}}, "r.yaml")


def test_local_scanner_uses_the_named_reader(tmp_path):
    run = F.write_eval_run(tmp_path, [F.Try(failed="grasp"), F.Try(seed=101)])
    scan = LocalScanner(spec(tmp_path, run), {"reader": "fake_recovery:read_messages"}).scan(tmp_path)
    assert [f.phase for f in scan.failures] == ["grasp"] and scan.counts["passed"] == 1


def test_local_scanner_defaults_to_the_rosbag2_reader(tmp_path):
    assert LocalScanner(spec(tmp_path, tmp_path), {}).reader == "recovery.bag_io:read_messages"


def test_scan_client_writes_the_scan_json(tmp_path, monkeypatch):
    run = F.write_eval_run(tmp_path, [F.Try(failed="grasp")])
    monkeypatch.setattr(scan_client, "read_messages", F.read_messages)
    s = spec(tmp_path, run)
    scan_client.main(["--spec", json.dumps(s.raw), "--run", str(run), "--out", str(tmp_path / "scan.json")])
    assert json.loads((tmp_path / "scan.json").read_text())["failures"][0]["phase"] == "grasp"


def test_container_scanner_command_and_mounts(tmp_path, monkeypatch):
    s = spec(tmp_path, tmp_path / "eval")
    sc, made = ContainerScanner(s, {"wall_s": 7}), []

    class Box(ClientContainer):
        def run(self, log, **kw):
            made.append((self, log, kw))
            (self.mounts[0] / "scan.json").write_text(json.dumps({"failures": [], "counts": {"passed": 2}}))
            return 0
    monkeypatch.setattr("recovery.plugins.scanners.container.ClientContainer", Box)
    (tmp_path / "out").mkdir()
    scan = sc.scan(tmp_path / "out")
    (box, log, kw), = made
    assert box.command[:3] == ["python3", "-m", "recovery.scan_client"] and json.loads(box.command[box.command.index("--spec") + 1]) == s.raw
    assert box.command[box.command.index("--run") + 1] == s.eval_run and box.command[-1] == str(tmp_path / "out" / "scan.json")
    assert [str(m) for m in box.mounts] == [str(tmp_path / "out"), s.eval_run] and kw == {"timeout": 7.0}
    assert box.name.startswith("rc-") and box.name.endswith("-scan") and scan.counts["passed"] == 2 and str(log).endswith("scan.log")
    argv = box.argv()                                   # the real container command: the sim image, the eval run mounted
    assert argv[:2] == ["docker", "run"] and "recovery.scan_client" in argv and f"{s.eval_run}:{s.eval_run}" in argv


def test_container_scanner_failures(tmp_path, monkeypatch):
    sc = ContainerScanner(spec(tmp_path, tmp_path), {})
    monkeypatch.setattr("recovery.plugins.scanners.container.ClientContainer.run", lambda self, log, **kw: 3)
    with pytest.raises(RecoveryError, match="exited with 3"):
        sc.scan(tmp_path)

    def no_docker(self, log, **kw):
        raise FileNotFoundError("docker")
    monkeypatch.setattr("recovery.plugins.scanners.container.ClientContainer.run", no_docker)
    with pytest.raises(RecoveryError, match="docker is not available"):
        sc.scan(tmp_path)


def test_bag_io_decodes_the_recorded_message_types():
    assert bag_io.decode("/evaluator/state", NS(data='{"t": 1}')) == '{"t": 1}'
    assert bag_io.decode("/joint_states", NS(name=["a", "b"], position=(0.1, 0.2))) == {"name": ["a", "b"], "position": [0.1, 0.2]}
    assert bag_io.decode("/world/w/model/fridge/joint_state", NS(name=["h"], position=[0.5])) == {"name": ["h"], "position": [0.5]}
    tf = lambda child, yaw: NS(child_frame_id=child, transform=NS(translation=NS(x=1.0, y=2.0, z=0.0),
                                                                  rotation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))))
    x, y, yaw = bag_io.decode("/model/mobile_panda/pose", NS(transforms=[tf("mobile_panda/base", 0.0), tf("mobile_panda", 1.2)]))
    assert (x, y) == (1.0, 2.0) and yaw == pytest.approx(1.2)
    assert bag_io.decode("/model/mobile_panda/pose", NS(transforms=[tf("other", 0.0)])) is None


def test_only_the_named_rollouts_are_scanned(tmp_path):
    run = F.write_eval_run(tmp_path, [F.Try(seed=100, failed="grasp"), F.Try(seed=101, failed="grasp"), F.Try(seed=102)])
    scanner = LocalScanner(spec(tmp_path, run), {"reader": "fake_recovery:read_messages"})
    scan = scanner.scan(tmp_path, {("mined", "101_r0")})
    assert [f.rollout for f in scan.failures] == ["101_r0"] and dict(scan.counts) == {"failed": 1}
    assert dict(scanner.scan(tmp_path, set()).counts) == {} and len(scanner.scan(tmp_path).failures) == 2


def test_the_container_scan_and_its_client_take_the_rollouts_as_only_arguments(tmp_path, monkeypatch):
    s = spec(tmp_path, tmp_path / "eval")
    command = ContainerScanner(s, {}).command(tmp_path, {("mined", "5_r1"), ("train", "2_r0")})
    assert command[-4:] == ["--only", "mined/5_r1", "--only", "train/2_r0"] and "--only" not in ContainerScanner(s, {}).command(tmp_path)
    run = F.write_eval_run(tmp_path, [F.Try(seed=100, failed="grasp"), F.Try(seed=101, failed="grasp")])
    monkeypatch.setattr(scan_client, "read_messages", F.read_messages)
    scan_client.main(["--spec", json.dumps(spec(tmp_path, run).raw), "--run", str(run), "--out", str(tmp_path / "s.json"),
                      "--only", "mined/100_r0"])
    assert [f["rollout"] for f in json.loads((tmp_path / "s.json").read_text())["failures"]] == ["100_r0"]
