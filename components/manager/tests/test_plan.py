"""Unit tests of the loop's state machine (plan.py): next step, skips, stop rules, inputs, held-out guard, outputs."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "components" / "manager" / "src")]
from manager import plan as P  # noqa: E402

from revolve_common import specs  # noqa: E402

SPEC = REPO / "specs" / "experiments" / "fridge_mini.yaml"


@pytest.fixture
def r():
    return specs.load_experiment(SPEC)


@pytest.fixture
def paths(tmp_path):
    return P.Paths(tmp_path / "runs" / "fridge_mini")


def finish(st, k, step, **outputs):
    st["turns"][k]["steps"][step].update(state="done", outputs=outputs)


def drive(st, max_turns, heldout_pass):
    """Walk the plan with every step succeeding; heldout_pass(k) = turn k's held-out verdict."""
    seen = []
    while True:
        a = P.next_action(st, max_turns)
        seen.append(a)
        if a[0] == "stop":
            return seen
        if a[0] == "turn":
            st["turns"].append(P.new_turn(a[1]))
        elif a[0] == "skip":
            st["turns"][a[1]]["steps"][a[2]]["state"] = "skipped"
        else:
            finish(st, a[1], a[2], **({"all_pass": heldout_pass(a[1])} if a[2] == "eval_heldout" else {}))


def test_order_and_stop_on_max_turns():
    st = P.new_state("x", SPEC)
    seen = drive(st, 3, lambda k: False)
    runs = [(a[1], a[2]) for a in seen if a[0] == "run"]
    assert runs[:6] == [(0, s) for s in ("generate", "starts", "dataset", "train", "eval_heldout", "eval_probe")]
    assert runs[6:11] == [(1, s) for s in ("recover", "dataset", "train", "eval_heldout", "eval_probe")]
    assert runs[11:] == [(2, s) for s in ("recover", "dataset", "train", "eval_heldout")]
    assert ("skip", 2, "eval_probe", "last turn: no recovery follows") in seen
    assert seen[-1] == ("stop", "max_turns") and len(st["turns"]) == 3


def test_stop_on_heldout_all_pass():
    st = P.new_state("x", SPEC)
    seen = drive(st, 5, lambda k: k == 1)
    assert seen[-1] == ("stop", "all_pass") and len(st["turns"]) == 2
    assert st["turns"][1]["steps"]["eval_probe"]["state"] == "skipped"


def test_single_turn_and_failed_step():
    st = P.new_state("x", SPEC)
    assert drive(st, 1, lambda k: False)[-1] == ("stop", "max_turns")
    st = P.new_state("x", SPEC)
    st["turns"][0]["steps"]["generate"]["state"] = "failed"
    assert P.next_action(st, 3) == ("stop", "failed")
    st["turns"][0]["steps"]["generate"]["state"] = "running"          # resume: the running step again
    assert P.next_action(st, 3) == ("run", 0, "generate")


def episode(d, **ep):
    d.mkdir(parents=True)
    (d / "episode.json").write_text(json.dumps(ep))
    return str(d)


def test_heldout_never_trains(r, paths):
    ok = [episode(paths.data() / "episodes" / "seed_0_a0", datagen={"seed": 0}),
          episode(paths.recovery(1) / "200_r0" / "episodes" / "ep_0", recovery={"rollout_seed": 200})]
    P.check_training_episodes(ok, r.experiment, paths)
    bad = episode(paths.recovery(1) / "100_r0" / "episodes" / "ep_0", recovery={"rollout_seed": 100})
    with pytest.raises(P.HeldoutLeak, match="held-out"):
        P.check_training_episodes([bad], r.experiment, paths)
    in_heldout = episode(paths.eval(0) / "heldout" / "x", datagen={"seed": 1})
    with pytest.raises(P.HeldoutLeak, match="held-out eval dir"):
        P.check_training_episodes([in_heldout], r.experiment, paths)
    st = P.new_state("x", SPEC)
    finish(st, 0, "generate", episodes=ok[:1] + [bad])
    with pytest.raises(P.HeldoutLeak):
        P.step_params(st, 0, "dataset", r, paths)


def test_recovery_inputs_are_probe_failures_only(r, paths):
    st = P.new_state("x", SPEC)
    rel = lambda p: str(p.relative_to(paths.exp.parents[1]))       # noqa: E731 (eval_summary dirs: repo-relative)
    finish(st, 0, "eval_probe", failures=[dict(seed=200, dir=rel(paths.eval(0) / "probe" / "200_r0"), failed_phase="open")])
    st["turns"].append(P.new_turn(1))
    p = P.step_params(st, 1, "recover", r, paths)
    assert p["argv"][p["argv"].index("--rollouts") + 1] == str(paths.eval(0) / "probe" / "200_r0")
    assert P.arg(p, "--phases") == "approach,grasp,open" and P.arg(p, "--episodes-per-failure") == "3"
    st["turns"][0]["steps"]["eval_probe"]["outputs"]["failures"] = [
        dict(seed=100, dir=rel(paths.eval(0) / "heldout" / "100_r0"), failed_phase="open")]
    with pytest.raises(P.HeldoutLeak):
        P.step_params(st, 1, "recover", r, paths)
    st["turns"][0]["steps"]["eval_probe"]["outputs"]["failures"] = []
    assert P.step_params(st, 1, "recover", r, paths) is None            # nothing failed: skipped


def test_dataset_and_train_chain(r, paths):
    st = P.new_state("x", SPEC)
    finish(st, 0, "generate", episodes=[episode(paths.data() / "episodes" / "seed_0_a0", datagen={"seed": 0})])
    p = P.step_params(st, 0, "dataset", r, paths)
    assert p["prev_dataset"] is None and p["turn"] == 0
    finish(st, 0, "dataset", dataset=str(paths.turn(0) / "dataset"))
    t = P.step_params(st, 0, "train", r, paths)
    assert t["prev"] is None and t["training"]["mode"] == "finetune_last" and t["training"]["backend"] == "gpu_machine"
    st["turns"].append(P.new_turn(1))
    finish(st, 1, "recover", episodes=[])
    assert P.step_params(st, 1, "dataset", r, paths) is None               # no new episodes: reuse turn 0's
    t = P.step_params(st, 1, "train", r, paths)
    assert t["dataset"] == str(paths.turn(0) / "dataset") and t["prev"] == str(paths.ckpt(0))
    finish(st, 1, "recover", episodes=[episode(paths.recovery(1) / "200_r0" / "episodes" / "ep_0", recovery={"rollout_seed": 200})])
    p = P.step_params(st, 1, "dataset", r, paths)
    assert p["prev_dataset"] == str(paths.turn(0) / "dataset")
    e = P.step_params(st, 1, "eval_heldout", r, paths)
    assert P.arg(e, "--seeds") == "100,101,102,103" and P.arg(e, "--ckpt") == str(paths.ckpt(1))
    assert P.arg(e, "--starts") == str(paths.starts)


def test_collect_eval(paths):
    d = paths.eval(0) / "probe"
    d.mkdir(parents=True)
    row = lambda ok, ph: dict(dir=f"runs/x/{ph}", first_try={"approach": "pass" if ok else "fail"},  # noqa: E731
                              passed_first_try=ok, failed_phase=None if ok else "approach", passed=ok)
    (d / "eval_summary.json").write_text(json.dumps(dict(all_pass=False, first_try_passed=1, rollouts=3, table={
        "200": {"r0": row(True, "a")}, "201": {"r0": row(False, "b")},
        "202": {"r0": dict(dir="runs/x/c", first_try=None, passed_first_try=False, failed_phase=None)}})))
    out, err = P.collect("eval_probe", dict(watch=str(d)), paths)
    assert err is None and out["all_pass"] is False
    assert out["failures"] == [dict(seed=201, dir="runs/x/b", failed_phase="approach")]   # stack errors are not recovered
    assert out["table"]["201"]["r0"] == {"approach": "fail"}
    assert P.collect("eval_heldout", dict(watch=str(paths.eval(0) / "heldout")), paths)[1] == "no eval_summary.json"


def test_summary(paths):
    st = P.new_state("x", SPEC)
    finish(st, 0, "eval_heldout", all_pass=True, first_try_passed=4, rollouts=4, table={"100": {"r0": {"approach": "pass"}}})
    st["turns"][0]["steps"]["train"]["attempts"] = [dict(state="failed", error="exit 1", wall_s=3.0),
                                                     dict(state="succeeded", wall_s=5.0)]
    P.save(st, paths)
    s = json.loads((paths.exp / "summary.json").read_text())
    assert s["turns"][0]["heldout"]["all_pass"] is True
    assert s["turns"][0]["steps"]["train"] == dict(state="pending", attempts=2, wall_s=8.0, error=None, reason=None)
    assert json.loads((paths.exp / "state.json").read_text())["spec_sha256"] == P.sha256(SPEC)
