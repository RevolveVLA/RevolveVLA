"""`revolve sim --task T --out DIR` into a directory left by a run killed mid-episode (mobile_panda, open_fridge_door):
DIR holds an unfinished bag episode_0_bag (one junk .mcap, no metadata.yaml) and no episode_0.json. One idle episode
(check_evaluator.py --mode idle) is started on /evaluator/start; checked:

  L1 the episode numbers continue past every episode file of DIR, bag directories included: the episode is number 1.
  L2 its bag is DIR/episode_1_bag, a closed rosbag2 bag with messages; the stale episode_0_bag is untouched and no
     other bag directory appears.
  L3 the result file is DIR/episode_1.json, with the result's episode and bag.
  L4 every path of the result is a host path; the gz server's console log of the episode (from its start to its end,
     not since the server started; may be empty) is copied to DIR/episode_1_gz.log before the result file is written.
  L5 /evaluator/state "values" holds every condition of the task (each phase's done / while, the merged sanity checks)
     before the episode too, null where it cannot be computed (the open phase's slip outside its phase).

Nothing is written at import or collection time: DIR is created by a module-scoped autouse fixture before the sim
starts, and deleted at the end of the module when every test passed (kept for debugging on a failure).
"""
import json
import os
import re
import shutil
import time

import pytest
from conftest import REPO, problem_lines, result, sh
from revolve_common import specs

ROBOTS = ["mobile_panda"]
TASK = "open_fridge_door"
OUT = REPO / "runs" / "test_simulator" / f"out_dir_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
STALE_BAG = OUT / "episode_0_bag"
STALE_FILE = STALE_BAG / "0_episode_0_bag_x.mcap"
STALE_BYTES = b"junk of a run killed mid-episode\x00\x01\x02"
SIM_ARGS = ["--task", TASK, "--no-autostart", "--out", str(OUT)]
GZ_STARTUP = "Gazebo Sim Server v"     # the gz server's startup line, written at launch before any episode


@pytest.fixture(scope="module", autouse=True)
def stale_out_dir(request):
    """DIR as a run killed mid-episode left it, before the sim starts (autouse: before the sim fixture); removed at
    the end of the module when every test passed."""
    # created here, not at import or collection time: episode_0_bag/ with its junk .mcap, no metadata.yaml,
    # no episode_0.json
    STALE_BAG.mkdir(parents=True)
    STALE_FILE.write_bytes(STALE_BYTES)
    failed = request.session.testsfailed
    yield OUT
    # every test of the module passed -> DIR deleted; on a failure it stays for debugging
    if request.session.testsfailed == failed:
        shutil.rmtree(OUT, ignore_errors=True)


def wait_file(path, timeout):
    """True once path exists (the result may be published a moment before the files are final)."""
    deadline = time.time() + timeout
    while not path.exists() and time.time() < deadline:
        time.sleep(0.5)
    return path.exists()


def container_path(host):
    """A host path under the repo -> the same path in the sim container (the repo at /repo)."""
    return f"/repo/{os.path.relpath(host, REPO)}"


def strings(obj):
    """Every string value of a JSON object, nested ones included."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


@pytest.fixture(scope="module")
def idle(sim):
    out = result(sim, "check_evaluator.py", f"--task {TASK} --mode idle", 240, sim["logs"] / "check_evaluator.log")
    assert out["result"] is not None, f"no /evaluator/result: see {sim['logs'] / 'check_evaluator.log'}"
    return out


def test_episode_number_continues_past_stale_bag(idle):
    # L1 next number past a stale bag directory (episode_0_bag, no episode_0.json): the episode started is 1
    ok, message = idle["start"]
    assert ok, f"/evaluator/start refused: {message!r}"
    assert message == "episode 1"
    assert idle["result"]["episode"] == 1


def test_bag_is_closed_and_stale_bag_untouched(sim, idle):
    bag = OUT / "episode_1_bag"
    # L2 the result's bag is DIR/episode_1_bag
    assert idle["result"]["bag"] == str(bag)
    # L2 a closed rosbag2 bag: metadata.yaml, `ros2 bag info` succeeds with a positive message count
    assert wait_file(bag / "metadata.yaml", 30), f"{bag} has no metadata.yaml (bag not closed)"
    r = sh(sim["name"], f"timeout 50 ros2 bag info {container_path(bag)}", 60)
    assert r.returncode == 0, f"ros2 bag info failed (rc {r.returncode}): {r.stdout}{r.stderr}"
    m = re.search(r"Messages:\s*(\d+)", r.stdout)
    assert m, f"no 'Messages:' line in ros2 bag info: {r.stdout}"
    assert int(m.group(1)) > 0
    # L2 the stale episode_0_bag untouched: the same single file, the same bytes, still no metadata.yaml
    assert sorted(p.name for p in STALE_BAG.iterdir()) == [STALE_FILE.name]
    assert STALE_FILE.read_bytes() == STALE_BYTES
    assert not (STALE_BAG / "metadata.yaml").exists()
    # L2 no other bag directory in DIR (e.g. no "episode_0_bag(1)")
    bags = {p.name for p in OUT.iterdir() if p.name.startswith("episode_") and "_bag" in p.name}
    assert bags == {"episode_0_bag", "episode_1_bag"}


def test_result_file(idle):
    rfile = OUT / "episode_1.json"
    # L3 the result's result_file is DIR/episode_1.json, on the host, with the result's episode and bag
    assert idle["result"]["result_file"] == str(rfile)
    assert wait_file(rfile, 30), f"no result file {rfile}"
    written = json.loads(rfile.read_text())
    assert written["episode"] == 1
    assert written["bag"] == idle["result"]["bag"]


def test_paths_are_host_paths_and_gz_log_copied(idle):
    rfile, gz_log = OUT / "episode_1.json", OUT / "episode_1_gz.log"
    # L4 the gz log copy is DIR/episode_1_gz.log
    assert idle["result"]["gz_log"] == str(gz_log)
    # L4 written before the result file: once the result file exists, the gz log copy exists too
    assert wait_file(rfile, 30), f"no result file {rfile}"
    assert gz_log.is_file(), f"result file written but no gz log copy {gz_log}"
    # L4 only what the gz server wrote during the episode (may be empty): not its startup line from before the episode
    text = gz_log.read_bytes().decode("utf-8", errors="replace")
    assert GZ_STARTUP not in text, f"{gz_log} holds the server's log since its start, not only the episode's"
    # L4 every path in a result written with --out is a host path (none in the container's /repo)
    for res in (idle["result"], json.loads(rfile.read_text())):
        container = [s for s in strings(res) if s == "/repo" or s.startswith("/repo/")]
        assert not container, f"container paths in a result: {container}"
    for key in ("bag", "result_file", "gz_log"):
        assert os.path.exists(idle["result"][key]), f"result {key} {idle['result'][key]} not on the host"


def test_state_values_hold_every_condition(idle):
    res = specs.resolve(REPO / "specs", TASK, assets=False)
    expected = {f"{p.name}.done[{i}]" for p in res.task.phases for i in range(len(p.done))}
    expected |= {f"{p.name}.while[{i}]" for p in res.task.phases for i in range(len(p.while_))}
    expected |= {f"sanity.{k}" for k, v in res.sanity.items() if v is not None}
    # L5 the state taken before the episode holds every condition of the task (done, while, sanity) and no other
    assert idle["state_before"] is not None, "no /evaluator/state before the episode"
    values = idle["state_before"]["values"]
    assert set(values) == expected
    # L5 the open phase's slip condition, outside its phase (before the episode): present, null
    open_phase = next(p for p in res.task.phases if p.name == "open")
    slip = next(f"open.while[{i}]" for i, c in enumerate(open_phase.while_) if c.metric == "slip")
    assert slip in values
    assert values[slip] is None


def test_no_problem_lines(sim, idle):
    # no gz / ROS problem lines in the launch log, the episode included
    assert not problem_lines(sim["launch_log"])
