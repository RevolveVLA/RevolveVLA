"""System test: `./revolve evaluate` as a subprocess on the host venv (CPU, no docker, no GPU, no network).

The plugins are the real ones where they run on CPU: the `heldout` / `train` sets (datagen's fake generator), the `local` / `hf`
checkpoint sources + relabel, the `smolvla` policy, the `fake` rollout (as `fake_eval_log`: + the state log c1/c2 read) and the judges
c0 / c1 / c2. `hf` runs against a stub huggingface_hub (fake_hf/): no download. Each test gets a root dir (`specs/` copied in, the
training spec, a checkpoint under runs/); paths in the specs are absolute, so the repo's runs/ is never written."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from datagen.core.state import State as Manifest

from evaluation.core import spec as S

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[2]
REVOLVE = REPO / "revolve"
TRAIN = {
    "task": "open_fridge_door", "robot": "mobile_panda", "generator": {"plugin": "fake_datagen"},
    "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
    "splits": {"train": {"target": 3, "pool": {"initial": [0, 9]},
                         "sampler": {"strategy": "stratified", "bins": {"fake_datagen.x": 3}}},
               "heldout": {"target": 2, "pool": {"initial": [20, 24]}, "record": False}},
}
HELDOUT = {"count": 3, "pool": {"initial": [100, 109], "replacement": [110, 119]}, "sampler": {"strategy": "sobol"},
           "min_distance": {"value": 0.1, "realised": False}}
SPEC = {
    "ckpt": {"source": "local", "path": "runs/ckpts/c"}, "policy": {"plugin": "smolvla"},
    "rollout": {"plugin": "fake_eval_log", "fail_seeds": [101], "push_seeds": [102], "closed_seeds": [0]},
    "judge": {"criterion": "c0"}, "rollouts": 1, "retries": 1,
    "sets": {"heldout": HELDOUT, "train": {"from": "train", "count": 2}},
}
# per judge: the final verdict of each config. 100: opens held; 101: a phase fails, the restored retry passes; 102: the door is
# pushed open (c1 yes, c2 no); train 0: the phases pass but the door stays shut (only c0 yes); train 1: opens held
EXPECTED = {"c0": {100: 1, 101: 1, 102: 1, 0: 1, 1: 1}, "c1": {100: 1, 101: 1, 102: 1, 0: 0, 1: 1},
            "c2": {100: 1, 101: 1, 102: 0, 0: 0, 1: 1}}


@pytest.fixture
def root(tmp_path):
    """A repo-like root: specs/ (task + robot contract), the training spec + manifest, a checkpoint to relabel."""
    shutil.copytree(REPO / "specs", tmp_path / "specs")
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(TRAIN | {"out": str(tmp_path / "train_run")}))
    ck = tmp_path / "runs" / "ckpts" / "c"
    ck.mkdir(parents=True)
    cams = {f"observation.images.camera{i}": {"type": "VISUAL", "shape": [3, 256, 256]} for i in (1, 2, 3)}
    (ck / "config.json").write_text(json.dumps({"input_features": cams}))
    (ck / "train_config.json").write_text(json.dumps({"rename_map": {
        f"observation.images.{n}": f"observation.images.camera{i}" for i, n in enumerate(("wrist", "overhead_left", "overhead_right"), 1)}}))
    (ck / "model.safetensors").write_bytes(b"\0" * 16)
    return tmp_path


def spec_of(root, **edits) -> dict:
    return {"train": str(root / "train.yaml"), "out": str(root / "runs" / "eval" / "sys")} | SPEC | edits


def training_manifest(root, entries=((0, -0.3), (1, 0.0), (2, 0.3)), touched=()):
    """<train out>/manifest.json: the used training configs (seed, x) with their start dirs; `touched` = extra seeds."""
    train = S.parse(spec_of(root), "eval.yaml", root).train
    m = Manifest.open(root / "train_run" / "state.json", train)
    rec = m.split("train")
    for seed, x in entries:
        start = root / "train_run" / "starts" / f"seed_{seed}"
        start.mkdir(parents=True)
        rec["sampled"].append({"split": "train", "seed": seed, "cell": [min(int((x + 0.5) * 3), 2)],
                               "values": {"fake_datagen.x": x, "fake_datagen.y": 1.5}})
        rec["used"].append({"config": rec["sampled"][-1], "generate": {"ok": True, "realised": {"fake_datagen.x": x}, "paths": {"check": str(start)}},
                            "record": {"ok": True}})
    rec["sampled"] += [{"split": "train", "seed": s, "cell": [0], "values": {}} for s in touched]
    m.save()


def evaluate(root, *args, spec=None, env=None, name="spec.yaml"):
    p = root / name
    p.write_text(yaml.safe_dump(spec or spec_of(root), sort_keys=False))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
        "EVAL_PLUGIN_PATH": str(TESTS), "DATAGEN_PLUGIN_PATH": str(REPO / "components" / "datagen" / "tests"),
        "FAKE_EVAL_LOG": str(root / "calls.log")} | (env or {})
    r = subprocess.run([sys.executable, str(REVOLVE), "evaluate", str(p), *args], capture_output=True, text=True, cwd=root, env=env)
    return r.returncode, r.stdout, r.stderr


def results(root, out="sys"):
    return json.loads((root / "runs" / "eval" / out / "results.json").read_text())


def calls(root):
    p = root / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


def verdicts(r):
    return {e["seed"]: int(e["passed"]) for s in r["sets"].values() for e in s["rollouts"].values()}


# ------------------------------------------------------------------ the pipeline end to end, per judge
@pytest.mark.parametrize("criterion", ["c0", pytest.param("c1", marks=pytest.mark.legacy), pytest.param("c2", marks=pytest.mark.legacy)])
def test_all_stages_with_each_judge(root, criterion):
    training_manifest(root)
    rc, out, err = evaluate(root, "all", spec=spec_of(root, judge={"criterion": criterion}))
    assert rc == 0, err
    assert "sample heldout: 3 configs" in out and "check heldout seeds_disjoint: PASS" in out and "check train" not in out
    r = results(root)
    assert verdicts(r) == EXPECTED[criterion]
    held = r["sets"]["heldout"]["rollouts"]
    assert sorted(held) == ["100_r0", "101_r0", "102_r0"] and sorted(r["sets"]["train"]["rollouts"]) == ["0_r0", "1_r0"]
    tries = held["101_r0"]["tries"]                                           # a retry from the failed phase, judged by its phase verdict
    assert [t["passed"] for t in tries] == [False, True] and tries[1]["restored_at"] == "release"
    assert tries[1]["detail"] == {"criterion": "c0" if criterion == "c0" else "c0_from_restore"} and tries[1]["dir"].endswith("101_r0/retry_1")
    detail = held["102_r0"]["tries"][0]["detail"]
    assert detail["criterion"] == criterion
    if criterion != "c0":                                                     # the log decides: pushed open = opened, not held
        assert detail["door_max_deg"] == 65.0 and detail["held_fraction"] == 0.0 and detail["opened"] is True
        assert r["sets"]["train"]["rollouts"]["0_r0"]["tries"][0]["detail"]["door_max_deg"] == 40.0
        assert (root / "runs" / "eval" / "sys" / "heldout" / "100_r0" / "state_log.npz").is_file()
    report = (root / "runs" / "eval" / "sys" / "report.md").read_text()
    assert "| 100 |" in report and "## " in report and "heldout" in report and "train" in report


def test_stages_one_by_one_and_set_filter(root):
    training_manifest(root)
    assert evaluate(root, "report")[0] == 0 and (root / "runs/eval/sys/report.md").is_file() and calls(root) == []
    rc, out, _ = evaluate(root, "sample", "--set", "train")
    assert rc == 0 and "sample train: 2 configs" in out and not (root / "runs/eval/sys/heldout").exists() and calls(root) == []
    assert evaluate(root, "sample")[0] == 0 and (root / "runs/eval/sys/heldout/manifest.json").is_file()
    rc, out, _ = evaluate(root, "check")
    assert rc == 0 and "check heldout in_range: PASS" in out and calls(root) == []
    assert evaluate(root, "rollout", "--set", "train")[0] == 0
    r = results(root)
    assert sorted(r["sets"]) == ["heldout", "train"] and not r["sets"]["heldout"].get("rollouts")
    assert len(r["sets"]["train"]["rollouts"]) == 2 and len(calls(root)) == 2
    assert evaluate(root, "rollout")[0] == 0 and len(calls(root)) == 2 + 3 + 1            # + 3 held-out first tries + 101's retry
    n = len(calls(root))
    assert evaluate(root, "all")[0] == 0 and len(calls(root)) == n                      # resume: nothing re-run


def test_a_set_filter_runs_only_that_set(root):
    training_manifest(root)
    rc, out, err = evaluate(root, "all", "--set", "train")
    assert rc == 0, err
    r = results(root)
    assert list(r["sets"]) == ["train"] and not (root / "runs/eval/sys/heldout").exists()
    assert sorted(verdicts(r)) == [0, 1]


def test_every_try_records_why_it_ended_and_the_limits_that_applied(root):
    training_manifest(root)
    ends = {100: "no_action", 101: "abort", 102: "wall_limit"}                # train seeds 0, 1: success (the fake's default)
    rollout = SPEC["rollout"] | {"end_reasons": ends, "rtf": 0.3, "budget_s": 20, "wall_safety": 3.0}
    rc, out, err = evaluate(root, "all", spec=spec_of(root, rollout=rollout, judge={"criterion": "c0"}))
    assert rc == 0, err
    r = results(root)
    held, train = r["sets"]["heldout"], r["sets"]["train"]
    assert [t["termination"] for t in held["rollouts"]["101_r0"]["tries"]] == ["abort", "abort"]
    assert [len(e["tries"]) for e in held["rollouts"].values()] == [1, 2, 1]
    assert held["rollouts"]["102_r0"]["tries"][0]["termination"] == "wall_limit" and held["metrics"]["errors"] == 1
    assert held["metrics"]["terminations"] == {"no_action": 1, "abort": 2, "wall_limit": 1}
    assert train["metrics"]["terminations"] == {"success": 2}
    assert held["rollout_policy"] == train["rollout_policy"] == {
        "budget_s": 20, "wait_s": 30.0, "after_success_s": 2.0, "wall_safety": 3.0, "setup_s": 240.0, "rtf": 0.3,
        "wall_limit_s": round((30 + 20 + 2) / 0.3 * 3.0, 1)}
    report = (root / "runs" / "eval" / "sys" / "report.md").read_text()
    assert "| termination | tries |\n|---|---|\n| no_action | 1 |\n| abort | 2 |\n| wall_limit | 1 |" in report
    assert "| success | 2 |" in report and "Rollout limits: budget_s 20, wait_s 30.0" in report and "wall_safety 3.0" in report


def test_dry_run_prints_the_plan_and_touches_nothing(root):
    rc, out, _ = evaluate(root, "all", "--dry-run", spec=spec_of(root, ckpt={"source": "hf", "repo": "org/name"}))
    assert rc == 0 and "[dry-run] set heldout (heldout): 3 held-out configs, sobol sampler, pool [100, 109] + [110, 119]" in out
    assert "ckpt: hf {'repo': 'org/name'} (nothing resolved)" in out and "judge c0" in out and "workers: by free GPU memory" in out
    assert "rollout limits: budget_s" in out and "rtf measured at run time" in out    # main's task has no `rollout:`: the yaml's limits (PRD 06)
    _, out, _ = evaluate(root, "all", "--dry-run", spec=spec_of(root, rollout={"plugin": "fake_eval_log", "rtf": 0.3, "budget_s": 20}))
    assert "rollout limits: budget_s 20, " in out and "rtf 0.3, wall limit" in out
    assert not (root / "runs" / "eval").exists() and not (root / "runs" / "ckpt_cache").exists() and calls(root) == []


# ------------------------------------------------------------------ the results do not depend on the worker count
def test_workers_1_and_3_give_identical_results(root, tmp_path_factory):
    training_manifest(root)
    spec = spec_of(root, rollouts=2, rollout=SPEC["rollout"] | {"delay_s": 0.2})
    texts = []
    for w in (1, 3):
        (root / "runs" / "eval" / "sys").parent.mkdir(parents=True, exist_ok=True)
        rc, out, err = evaluate(root, "all", "--workers", str(w), spec=spec)
        assert rc == 0, err
        texts.append((root / "runs/eval/sys/results.json").read_text())
        texts.append((root / "runs/eval/sys/report.md").read_text())
        shutil.rmtree(root / "runs" / "eval" / "sys")
        (root / "calls.log").unlink()
    assert texts[0] == texts[2] and texts[1] == texts[3]
    assert len(json.loads(texts[0])["sets"]["heldout"]["rollouts"]) == 6


def test_workers_flag_runs_rollouts_in_parallel(root):
    training_manifest(root)
    spec = spec_of(root, rollouts=2, retries=0, rollout=SPEC["rollout"] | {"delay_s": 0.6, "fail_seeds": []},
                   sets={"heldout": HELDOUT})
    t0 = time.time()
    assert evaluate(root, "rollout", "--workers", "6", spec=spec)[0] == 0
    assert time.time() - t0 < 6 * 0.6 + 1                                              # 6 rollouts of 0.6 s: well under serial + startup


# ------------------------------------------------------------------ interrupt and resume
def test_sigint_exits_130_and_the_rerun_completes_the_rest(root):
    training_manifest(root)
    spec = spec_of(root, rollouts=3, rollout=SPEC["rollout"] | {"delay_s": 0.7}, run={"workers": 1})
    (root / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
        "EVAL_PLUGIN_PATH": str(TESTS), "DATAGEN_PLUGIN_PATH": str(REPO / "components" / "datagen" / "tests"),
        "FAKE_EVAL_LOG": str(root / "calls.log")}
    p = subprocess.Popen([sys.executable, str(REVOLVE), "evaluate", str(root / "spec.yaml"), "rollout"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=root, env=env)
    for _ in range(300):
        if len(calls(root)) >= 3:
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGINT)
    out, err = p.communicate(timeout=60)
    assert p.returncode == 130 and "Traceback" not in err
    done = set(calls(root))
    assert 3 <= len(done) < 15
    rc, _, err = evaluate(root, "rollout", spec=spec)
    assert rc == 0, err
    again = calls(root)[len(done):]
    assert not done & set(again)                                                      # what finished before the signal is kept
    assert len(verdicts(results(root))) == 5 and len(results(root)["sets"]["heldout"]["rollouts"]) == 9
    assert (root / "runs/eval/sys/results.json").is_file()


# ------------------------------------------------------------------ failures: exit codes
def test_a_check_that_fails_exits_1(root):
    training_manifest(root, touched=(105,))                                           # the training run touched a held-out pool seed
    rc, out, err = evaluate(root, "check")
    assert rc == 1 and "check heldout seeds_disjoint: FAIL (seeds also used by training [105])" in out and "Traceback" not in err
    assert not results(root)["sets"]["heldout"]["checks"]["seeds_disjoint"]["ok"]


def test_held_out_starts_too_close_to_every_training_config_exit_1(root):
    train = TRAIN | {"generator": {"plugin": "fake_datagen", "realised": {s: 0.0 for s in range(100, 120)}}}
    (root / "train.yaml").write_text(yaml.safe_dump(train | {"out": str(root / "train_run")}))
    training_manifest(root, entries=((0, 0.0),))
    spec = spec_of(root, sets={"heldout": HELDOUT | {"count": 2, "min_distance": {"value": 0.1, "realised": True}}})
    rc, out, err = evaluate(root, "sample", spec=spec)
    assert rc == 1 and "Traceback" not in err and err.startswith("evaluate: ")        # the pool ran out of starts away from training


def test_a_realised_distance_needs_the_training_manifest(root):
    spec = spec_of(root, sets={"heldout": HELDOUT | {"min_distance": {"value": 0.1, "realised": True}}})
    rc, _, err = evaluate(root, "check", spec=spec)
    assert rc == 1 and "needs the training manifest" in err and not (root / "runs/eval/sys/heldout/starts").exists()
    rc, out, _ = evaluate(root, "check", spec=spec_of(root, sets={"heldout": HELDOUT}))     # `realised: false` works without it (sampled-only)
    assert rc == 0 and "check heldout training_configs: PASS (sampled-only" in out


def test_a_bad_spec_exits_2_naming_the_key(root):
    rc, _, err = evaluate(root, "all", spec=spec_of(root, retrys=1))
    assert rc == 2 and "retrys: unknown key" in err and "Traceback" not in err
    assert evaluate(root, "all", spec=spec_of(root) | {"train": str(root / "missing.yaml")})[0] == 2
    rc, _, err = evaluate(root, "all", spec=spec_of(root, sets={"h": {"count": 1, "pool": {"initial": [1, 2]}}}))
    assert rc == 2 and "sets.h.pool" in err
    rc, _, err = evaluate(root, "all", spec=spec_of(root, judge={"criterion": "c2"}))           # main's tasks have no `success:` rule
    assert rc == 2 and "judge.criterion: c2 needs a `success:` block in task open_fridge_door" in err


def test_a_changed_spec_refuses_the_existing_out_and_run_does_not_matter(root):
    training_manifest(root)
    assert evaluate(root, "all")[0] == 0
    rc, _, err = evaluate(root, "all", spec=spec_of(root, rollouts=2))
    assert rc == 1 and "another spec" in err
    assert evaluate(root, "all", spec=spec_of(root, run={"workers": 2}))[0] == 0


def test_a_missing_ckpt_or_plugin_is_one_line_exit_1(root):
    training_manifest(root)
    rc, _, err = evaluate(root, "rollout", spec=spec_of(root, ckpt={"source": "local", "path": "runs/ckpts/nope"}))
    assert rc == 1 and "ckpt runs/ckpts/nope: no config.json" in err and "Traceback" not in err
    rc, _, err = evaluate(root, "rollout", spec=spec_of(root, rollout={"plugin": "no_such_rollout"}))
    assert rc == 1 and "no rollout plugin 'no_such_rollout'" in err


# ------------------------------------------------------------------ the Hugging Face checkpoint, without a download
def test_hf_checkpoint_with_a_stubbed_download(root):
    training_manifest(root)
    hf = {"source": "hf", "repo": "org/name", "revision": "main", "subfolder": "checkpoints/last/pretrained_model"}
    log = root / "hf.log"
    env = {"EVAL_PLUGIN_PATH": os.pathsep.join([str(TESTS / "fake_hf"), str(TESTS)]), "FAKE_HF_LOG": str(log)}
    spec = spec_of(root, ckpt=hf, sets={"heldout": HELDOUT}, judge={"criterion": "c0"})
    rc, out, err = evaluate(root, "all", spec=spec, env=env)
    assert rc == 0, err
    sha = "0123456789abcdef0123456789abcdef01234567"
    ckpt = results(root)["ckpt"]
    assert ckpt["origin"]["source"] == "hf" and ckpt["origin"]["commit"] == sha and ckpt["origin"]["repo"] == "org/name"
    cache = root / "runs" / "ckpt_cache" / "hf" / "org__name" / sha / "checkpoints" / "last" / "pretrained_model"
    assert (cache / "config.json").is_file() and ckpt["path"] == str(cache)
    assert log.read_text().splitlines() == ["model_info org/name main",
                                            f"snapshot_download org/name {sha} ['checkpoints/last/pretrained_model/*']"]
    shutil.rmtree(root / "runs" / "eval")
    assert evaluate(root, "all", spec=spec, env=env)[0] == 0
    assert log.read_text().count("snapshot_download") == 1 and log.read_text().count("model_info") == 2   # cached: no second download


# ------------------------------------------------------------------ the repo's specs (GPU runs): valid, no training manifest needed
@pytest.mark.parametrize("name", ["fridge_smoke_eval", "fridge_base50_30k_nomanifest"])
def test_the_gpu_specs_hold_out_without_a_training_manifest(name):
    spec = S.load(REPO / "specs" / "eval" / f"{name}.yaml", REPO)
    assert spec.sets["heldout"].min_distance.realised is False and list(spec.sets) == ["heldout"]
    assert spec.judge.plugin == "c2" and spec.rollout.plugin == "docker"
    rc = subprocess.run([sys.executable, str(REVOLVE), "evaluate", f"specs/eval/{name}.yaml", "all", "--dry-run"],
                        capture_output=True, text=True, cwd=REPO)
    assert rc.returncode == 0, rc.stderr
