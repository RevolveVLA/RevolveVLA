"""The manager loop against stub components (stub_steps.py): the manager container, step containers, wrapper HTTP,
labels, stall kill, retry, kill + resume are real; the components' work is scripted. Needs revolve-manager
(./revolve build manager) and docker. ~3 min."""
import json
import shutil
import sys
import time
from pathlib import Path

import docker
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "components" / "manager" / "src")]
from manager import cli, loop  # noqa: E402

ROOT = REPO / "runs" / "manager_tests"
HELD = {100, 101}


@pytest.fixture(scope="module", autouse=True)
def image():
    try:
        docker.from_env().images.get(loop.IMAGE)
    except docker.errors.ImageNotFound:
        pytest.skip("no revolve-manager image: ./revolve build manager")


def experiment(name, max_turns, script, monkeypatch, stall=6):
    specs = ROOT / "specs"
    if not specs.is_dir():
        for k in ("tasks", "robots", "scenes"):
            shutil.copytree(REPO / "specs" / k, specs / k)
    (specs / "experiments").mkdir(parents=True, exist_ok=True)
    spec = specs / "experiments" / f"{name}.yaml"
    spec.write_text(f"""name: {name}
task: open_fridge_door
max_turns: {max_turns}
seeds: {{train: [0, 1], probe: [200, 201], heldout: [100, 101]}}
datagen: {{episodes_per_seed: 1, workers: 1}}
recovery: {{episodes_per_failure: 2, phases: [approach, grasp, open]}}
training: {{policy: smolvla, base: lerobot/smolvla_base, mode: finetune_last, steps: 20, batch_size: 2, backend: local}}
eval: {{rollouts_per_seed: 1, retries: 0}}
stall_timeout_s: {stall}
""")
    cli.stop(name) if (REPO / "runs" / name).exists() else None
    shutil.rmtree(REPO / "runs" / name, ignore_errors=True)
    stub = ROOT / f"{name}.stub.json"
    stub.write_text(json.dumps(script))
    monkeypatch.setenv("REVOLVE_STUB", str(stub))
    monkeypatch.setenv("REVOLVE_POLL_S", "0.5")
    return spec


def state(name):
    return json.loads((REPO / "runs" / name / "state.json").read_text())


def wait(pred, name, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = REPO / "runs" / name / "state.json"
        if f.is_file():
            try:
                st = state(name)
            except json.JSONDecodeError:
                st = None
            if st and pred(st):
                return st
        time.sleep(0.5)
    log = REPO / "runs" / name / "manager.log"
    raise AssertionError(f"timeout; manager.log tail:\n{log.read_text()[-3000:] if log.is_file() else '-'}")


def finished(st):
    return st["status"] in ("done", "failed")


def calls(name):
    return [json.loads(x) for x in (REPO / "runs" / name / "stub_calls.jsonl").read_text().splitlines()]


def leftovers(name):
    return docker.from_env().containers.list(all=True, filters={"label": f"revolve.exp={name}"})


def seed_of(d):
    ep = json.loads((Path(d) / "episode.json").read_text())
    return ep["recovery"]["rollout_seed"] if "recovery" in ep else ep["datagen"]["seed"]


@pytest.fixture
def cleanup():
    names = []
    yield names
    for n in names:
        cli.stop(n)


def test_three_turns_recovery_kill_resume_all_pass(monkeypatch, capsys, cleanup):
    name = "mt_loop"
    cleanup.append(name)
    spec = experiment(name, 4, dict(durations={"t1/train": 10}, verdicts={
        "0": {"heldout": {"100": "open"}, "probe": {"200": "approach", "201": "open"}},
        "1": {"heldout": {"101": "grasp"}, "probe": {"200": "grasp"}},
        "2": {}}), monkeypatch)
    assert cli.start(str(spec), resume=False) == 0
    with pytest.raises(SystemExit):
        cli.start(str(spec), resume=False)              # one manager per experiment

    def t1_training(st):
        if len(st["turns"]) < 2:
            return False
        a = st["turns"][1]["steps"]["train"]["attempts"]
        return bool(a) and "step" in (a[-1].get("progress") or {})
    st = wait(t1_training, name)
    step = st["turns"][1]["steps"]["train"]["attempts"][0]["container"]
    capsys.readouterr()
    cli.status(name)
    out = capsys.readouterr().out
    assert "live train" in out and "'loss'" in out, out
    assert "heldout: all_pass False (1/2 first try)" in out and "100_r0" in out, out

    dk = docker.from_env()
    dk.containers.get(cli.manager_name(name)).kill()    # the manager dies mid-job; the step keeps running
    time.sleep(1)
    assert dk.containers.get(step).status == "running"
    assert cli.start(name, resume=True) == 0
    st = wait(finished, name)
    log = (REPO / "runs" / name / "manager.log").read_text()
    assert st["status"] == "done" and st["stop_reason"] == "all_pass", log[-3000:]
    assert f"re-attaching to {step}" in log
    assert len(st["turns"]) == 3
    # nothing redone: each (turn, step) ran exactly once, the killed-over training too
    ran = [(c["turn"], c["step"]) for c in calls(name)]
    assert len(ran) == len(set(ran)), ran
    assert st["turns"][1]["steps"]["train"]["attempts"][0]["container"] == step
    assert len(st["turns"][1]["steps"]["train"]["attempts"]) == 1
    # recovery from the probe failures; eval_probe skipped on the all-pass turn
    t1, t2 = st["turns"][1]["steps"], st["turns"][2]["steps"]
    assert len(t1["recover"]["outputs"]["episodes"]) == 4 and len(t2["recover"]["outputs"]["episodes"]) == 2
    assert t2["eval_probe"]["state"] == "skipped"
    # held-out seeds never reach training; turn k fine-tunes from turn k-1
    for k in range(3):
        d = REPO / "runs" / name / f"turn_{k}"
        eps = json.loads((d / "dataset" / "stub_episodes.json").read_text())
        seeds = {seed_of(e) for e in eps}
        assert not seeds & HELD, (k, seeds)
        assert seeds == ({0, 1} if k == 0 else {0, 1, 200, 201})
        t = json.loads((d / "ckpt" / "train.json").read_text())
        assert t["prev"] == (None if k == 0 else str(REPO / "runs" / name / f"turn_{k - 1}" / "ckpt"))
        assert t["dataset"] == str(d / "dataset")
    s = json.loads((REPO / "runs" / name / "summary.json").read_text())
    assert s["status"] == "done" and s["turns"][2]["heldout"]["all_pass"] is True
    assert s["turns"][0]["heldout"]["table"]["100"]["r0"]["open"] == "fail"
    capsys.readouterr()
    cli.status(name)
    assert "stop: all_pass" in capsys.readouterr().out
    time.sleep(1)
    assert not [c.name for c in leftovers(name)]


def test_stall_kill_retry_and_max_turns(monkeypatch, cleanup):
    name = "mt_stall"
    cleanup.append(name)
    spec = experiment(name, 2, dict(hang={"t0/dataset": 1}, fail={"t1/train": 1}, verdicts={
        "0": {"heldout": {"100": "approach"}, "probe": {"200": "open"}},
        "1": {"heldout": {"100": "approach"}}}), monkeypatch, stall=5)
    assert cli.start(str(spec), resume=False) == 0
    st = wait(finished, name)
    log = (REPO / "runs" / name / "manager.log").read_text()
    assert st["status"] == "done" and st["stop_reason"] == "max_turns", log[-3000:]
    ds = st["turns"][0]["steps"]["dataset"]["attempts"]
    assert [a["state"] for a in ds] == ["stalled", "succeeded"], ds
    stalled = ds[0]["container"]
    dk = docker.from_env()
    assert loop.get(dk, stalled) is None and loop.get(dk, f"{stalled}-child") is None   # step + its child removed
    tr = st["turns"][1]["steps"]["train"]["attempts"]
    assert [a["state"] for a in tr] == ["failed", "succeeded"] and tr[0]["error"] == "exit 1"
    assert st["turns"][1]["steps"]["eval_probe"]["state"] == "skipped"     # last turn
    assert len(st["turns"]) == 2


def test_failed_twice_stops_with_error(monkeypatch, cleanup):
    name = "mt_fail"
    cleanup.append(name)
    spec = experiment(name, 3, dict(fail={"t0/train": 2}), monkeypatch)
    assert cli.start(str(spec), resume=False) == 0
    st = wait(finished, name)
    assert st["status"] == "failed" and st["stop_reason"] == "failed"
    assert [a["state"] for a in st["turns"][0]["steps"]["train"]["attempts"]] == ["failed", "failed"]
    s = json.loads((REPO / "runs" / name / "summary.json").read_text())
    assert "turn 0 train: failed twice" in s["error"]
    assert st["turns"][0]["steps"]["eval_heldout"]["state"] == "pending"
    assert [c["step"] for c in calls(name)].count("train") == 2
