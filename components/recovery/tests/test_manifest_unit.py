"""core/manifest.py: the DATAGEN-v2.md 8.4 entries of the job logs with the recovery block, `keep`, the reset records."""
import json

from recovery.core import manifest as M
from recovery.core import spec as S
from recovery.core.failures import Failure
from recovery.core.restart import Restart

RESTART = Restart(10.0, (-1.5, 1.7, 1.5), (0.1,) * 7, 0.0)


def spec(keep="successes", out="o"):
    return S.parse({"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": "/runs/eval"}, "out": out, "keep": keep,
                    "phases": {"generator": {"plugin": "x"}, "quota": {"grasp": 1}, "restart_from": "before_failure"}}, "r.yaml")


def job(tmp_path, tries, resets=None, seed=4):
    """A job dir with a log of `tries` ([(episode, passed)]) on the sim folder `<out>/sims/rc-0` and its result files."""
    out = tmp_path / "out"
    sim, d = out / "sims" / "rc-0", out / "grasp" / "mined_100_r0_t0"
    sim.mkdir(parents=True), d.mkdir(parents=True)
    for n, passed in tries:
        (sim / f"episode_{n}.json").write_text(json.dumps({"passed": passed}))
        (sim / f"episode_{n}_bag").mkdir()
    (d / "datagen.json").write_text(json.dumps({"seed": seed, "faults": [], "tries": [
        {"attempt": i, "episode": n, "passed": p, "reason": None, "wall_s": 1.5, "expert": "fake_v1", "notes": "", "sim": str(sim)}
        for i, (n, p) in enumerate(tries)]}))
    if resets is not None:
        (d / "resets.json").write_text(json.dumps(resets))
    f = Failure("mined_100_r0_t0", "mined", "100_r0", 0, "/runs/eval/scene.yaml", "grasp", "verdict", "grasp:timeout_s", 11.0, 6.0,
                "/runs/eval/bag", "res", RESTART)
    return out, f, d


def test_entry_shape(tmp_path):
    out, f, d = job(tmp_path, [(2, True)], resets=[{"attempt": 0, "mode": "warm", "ok": True, "base_err_m": 0.01}])
    (e,) = M.entries(out, d, f, spec())
    assert {k: e[k] for k in ("seed", "scene", "robot", "task", "episode", "result", "bag", "passed")} == {
        "seed": 4, "scene": "grasp/mined_100_r0_t0/scene.yaml", "robot": "mobile_panda", "task": "open_fridge_door", "episode": 2,
        "result": "sims/rc-0/episode_2.json", "bag": "sims/rc-0/episode_2_bag", "passed": True}
    assert e["datagen"] == {"expert": "fake_v1", "attempt": 0, "wall_s": 1.5}
    assert e["recovery"] == {
        "kind": "phase", "from_rollout": {"run": "/runs/eval", "set": "mined", "rollout": "100_r0", "try": 0,
                                          "scene": "/runs/eval/scene.yaml", "bag": "/runs/eval/bag"},
        "phase": "grasp", "trigger": "verdict", "reason": "grasp:timeout_s", "t_restart": 10.0, "restart_from": "before_failure",
        "restart_state": {"base": (-1.5, 1.7, 1.5), "arm": (0.1,) * 7, "object_joint": 0.0},
        "reset": {"mode": "warm", "ok": True, "base_err_m": 0.01}, "segment": None}


def test_keep_successes_or_all(tmp_path):
    out, f, d = job(tmp_path, [(0, False), (1, True)])
    assert [e["episode"] for e in M.entries(out, d, f, spec("successes"))] == [1]
    assert [(e["episode"], e["passed"]) for e in M.entries(out, d, f, spec("all"))] == [(0, False), (1, True)]


def test_reset_records():
    ok, miss = {"attempt": 0, "mode": "warm", "ok": True}, {"attempt": 0, "mode": "warm", "ok": False, "reason": "off"}
    assert M.reset_of([], 0) == {"mode": "relaunch"}
    assert M.reset_of([ok], 0) == {"mode": "warm", "ok": True}
    assert M.reset_of([ok], 1) == {"mode": "relaunch"}
    assert M.reset_of([miss], 0) == {"mode": "relaunch", "fallback": {"mode": "warm", "ok": False, "reason": "off"}}


def test_a_try_whose_result_file_is_gone_is_left_out(tmp_path):
    out, f, d = job(tmp_path, [(0, True), (1, True)])
    (out / "sims" / "rc-0" / "episode_0.json").unlink()
    assert [e["episode"] for e in M.entries(out, d, f, spec())] == [1]


def test_build_counts_and_order(tmp_path):
    out, f, d = job(tmp_path, [(0, False), (1, True)])
    data = M.build(out, spec("all"), [(f, d)], created="2026-11-24T00:00:00Z")
    assert data["counts"] == {"jobs": 1, "scenes": 1, "attempted": 2, "passed": 1} and data["created"] == "2026-11-24T00:00:00Z"
    assert (data["task"], data["robot"]) == ("open_fridge_door", "mobile_panda") and M.build(out, spec(), [])["episodes"] == []
    M.write(out / "m.json", data)
    assert M.created(out / "m.json") == "2026-11-24T00:00:00Z" and M.created(out / "none.json") is None
