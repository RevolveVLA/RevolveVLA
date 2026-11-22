"""Evaluation tests. test_evaluation_unit.py: stubs, no docker. test_evaluation.py: real rollouts (sim +
serve + Rosetta's runner on per-rollout internal networks), all started at once by one module fixture:
  oracle    evaluate(policy="oracle") on seed 0: the start's own expert actions through the ROS 2 path
            (client container -> DDS over the internal network -> ros_relay -> closed_loop_system) and the
            live phase validator: every phase must pass;
  heldout   evaluate.py under revolve_common.wrapper (host process, /status polled on 127.0.0.1) with the
            test checkpoint on seeds 0 and 1, retries 1, 2 at a time, budget_s 6 (short: an untrained policy).

Fixtures are not committed binaries; made on the first run into runs/test_cache/evaluation/ and reused (delete
it to regenerate): the checkpoint (20 steps of lerobot-train from lerobot/smolvla_base on the training tests'
cached turn_0 dataset), the starts of seeds 0 (the recovery tests' cached datagen episode) and 1 (planned +
checked here: starts.py). Outputs: runs/evaluation_tests/<stamp>/; numbers.json there sums up.
test_evaluation_so101.py: an oracle rollout of a second robot (the mobile SO-101 on the drawer, see its doc).

Needs: `./revolve build evaluation` (+ simulator, training, dataset), the kitchen assets in assets,
the caches of `./revolve test recovery` (datagen seed 0) and `./revolve test training` (turn_0 dataset), cache/hf
with lerobot/smolvla_base + SmolVLM2 (the training tests download them), a free GPU (~1.5 GB per rollout).
"""
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


import pytest

os.environ["REVOLVE_ROBOT"] = "mobile_panda"    # the robot of every job here (the fixtures are its runs)

COMP = Path(__file__).resolve().parents[1]
REPO = COMP.parents[1]
sys.path[:0] = [str(COMP), str(COMP / "src" / "evaluation"), str(COMP / "src"), str(REPO / "components" / "datagen" / "src")]
sys.path.append(str(REPO / "components" / "datagen" / "tests"))      # fake_datagen: the fake training generator
import legacy  # noqa: E402

LEGACY = {      # old simulator rollout / state-log judges / Rosetta contract relabel / starts (TRIAGE.md); PRD 07 deletes them
    "test_backcompat_unit.py": None, "test_cli_gpu.py": None, "test_evaluation.py": None, "test_evaluation_so101.py": None,
    "test_evaluation_unit.py": None, "test_success_unit.py": None,
    "test_ckpts_unit.py": {"test_served_ckpt_unchanged_when_it_matches_the_contract",     # need robots/*/contract.yaml (gone)
                           "test_served_ckpt_renames_and_relabels_an_external_checkpoint", "test_served_ckpt_relabels_the_shape_alone",
                           "test_served_ckpt_refuses_an_unknown_camera", "test_relabel_wraps_a_source_and_serves_under_out"},
    "test_judges_unit.py": {"test_verdicts_on_state_logs", "test_a_missing_log_fails_with_the_reason_and_a_task_without_rule_is_an_error",
                            "test_default_judge_follows_the_tasks_success_rule", "test_the_success_block_changes_no_datagen_spec_hash"},
    "test_termination_unit.py": {"test_rollout_policy_agrees_with_rollout_py_and_the_shipped_task",
                                 "test_docker_uses_the_measured_rtf_unless_the_yaml_gives_one",
                                 "test_docker_passes_the_wall_limit_params_and_names_them"},
    "test_rollout_unit.py": {"test_runner_params_yaml_default_and_override", "test_runner_params_yaml_aggregate",
                             "test_docker_passes_the_rollout_run_kwargs", "test_docker_sets_the_robot_env_for_rollout_py",
                             "test_docker_default_workers_is_the_gpu_default"},
    "test_spec_unit.py": {"test_the_repo_example_loads", "test_aggregate_names_match_rollout_py"},
    "test_trainings_unit.py": {"test_the_real_fridge_base50_run_loads"},
    "test_cli_unit.py": {"test_revolve_evaluate_dry_run_on_the_repo_spec"},
    "test_cli_system.py": {"test_every_try_records_why_it_ended_and_the_limits_that_applied",
                           "test_the_gpu_specs_hold_out_without_a_training_manifest"},
}
collect_ignore = legacy.ignored(LEGACY)
CACHE = REPO / "runs" / "test_cache" / "evaluation"
RUN = REPO / "runs" / "evaluation_tests" / time.strftime("%Y%m%d_%H%M%S")
TASK = "open_fridge_door"
DATAGEN = REPO / "runs" / "test_cache" / "recovery" / "datagen" / TASK
DATASET = REPO / "runs" / "test_cache" / "training" / "turn_0" / "dataset"
SEEDS = [0, 1]
BUDGET_S = 6.0


def load(p):
    return json.loads(Path(p).read_text())


def preflight():
    import rollout as R
    imgs = [R.SIM_ROS_IMAGE, R.train_launch.IMAGE, R.data_launch.IMAGE]
    missing = [i for i in imgs if subprocess.run(["docker", "image", "inspect", i], capture_output=True).returncode]
    missing += [str(p) for p in (R.sim_launch.INSTALL / "setup.bash", R.data_launch.INSTALL / "BUILD_OK",
                                 REPO / "assets/generated_partnet/fridge/model.sdf",
                                 DATAGEN / "manifest.json") if not p.is_file()]
    if missing:
        pytest.fail(f"not built / missing: {missing} (./revolve build evaluation; ./revolve test recovery)", pytrace=False)


def checkpoint():
    """runs/test_cache/evaluation/ckpt: a 20-step SmolVLA checkpoint (training's launcher, local GPU)."""
    ck = CACHE / "ckpt"
    if (ck / "model.safetensors").is_file():
        return ck
    if not (DATASET / "meta" / "info.json").is_file():
        pytest.fail(f"no {DATASET}: run ./revolve test training first", pytrace=False)
    import rollout as R
    res = R.train_launch.train(DATASET, ck, mode="from_base", base="lerobot/smolvla_base", steps=20, batch_size=2,
                               backend="local", job_dir=CACHE / "jobs" / "train_ckpt")
    assert res["exit"] == 0 and (ck / "model.safetensors").is_file(), res
    return ck


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def poll(port, seen, proc):
    import httpx
    while proc.poll() is None:
        try:
            s = httpx.get(f"http://127.0.0.1:{port}/status", timeout=2).json()
            seen.append(dict(state=s["state"], progress=s["progress"]))
        except Exception:
            pass
        time.sleep(2.0)


@pytest.fixture(scope="session")
def runs():
    preflight()
    import evaluate as E
    import starts as S
    t0 = time.time()
    ck = checkpoint()
    starts = S.resolve(TASK, SEEDS, CACHE / "starts", sources=[DATAGEN])
    t_fix = time.time() - t0
    out = dict(ckpt=ck, starts=starts, fixtures_s=round(t_fix, 1), dir=RUN)
    RUN.mkdir(parents=True, exist_ok=True)
    port, seen = free_port(), []
    job = RUN / "jobs" / "evaluate_heldout"
    job.mkdir(parents=True)
    cmd = [sys.executable, "-m", "revolve_common.wrapper", "--job-dir", str(job), "--host", "127.0.0.1", "--port", str(port),
           "--parser", "eval_progress:parse", "--linger-s", "0", "--", sys.executable, str(COMP / "evaluate.py"),
           "--task", TASK, "--ckpt", str(ck), "--seeds", ",".join(map(str, SEEDS)), "--set", "heldout", "--out", str(RUN),
           "--starts", str(CACHE / "starts"), "--source", str(DATAGEN), "--retries", "1", "--parallel", "2",
           "--budget-s", str(BUDGET_S)]
    env = dict(__import__("os").environ, PYTHONPATH=f"{REPO / 'common'}:{COMP / 'src' / 'evaluation'}")
    t1 = time.time()
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(RUN / "evaluate_heldout.log", "w"), stderr=subprocess.STDOUT)
    th = threading.Thread(target=poll, args=(port, seen, proc), daemon=True)
    th.start()
    oracle = E.evaluate(TASK, None, [0], RUN, "oracle", CACHE / "starts", policy="oracle", parallel=1, sources=[DATAGEN])
    out["oracle_wall_s"] = round(time.time() - t1, 1)
    proc.wait(timeout=1800)
    th.join(timeout=10)
    out.update(oracle=oracle, heldout=load(RUN / "heldout" / "eval_summary.json"), wrapper_rc=proc.returncode,
               wrapper_status=seen, wrapper_result=load(job / "result.json"), evals_wall_s=round(time.time() - t1, 1))
    (RUN / "numbers.json").write_text(json.dumps(dict(
        fixtures_s=out["fixtures_s"], evals_wall_s=out["evals_wall_s"], oracle_wall_s=out["oracle_wall_s"],
        oracle={k: oracle[k] for k in ("all_pass", "rtf", "rollout_wall_s", "gpu_memory")},
        heldout={k: out["heldout"][k] for k in ("first_try_passed", "rollouts", "rtf", "rollout_wall_s", "gpu_memory",
                                                "eval_wall_s")},
        table={s: {r: dict(first_try=v["first_try"], tries=v["tries"]) for r, v in t.items()}
               for s, t in out["heldout"]["table"].items()}), indent=1))
    return out


def _module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a free GPU, docker and the built images; set EVAL_GPU=1 to run")
    config.addinivalue_line("markers", "legacy: tests of code the new simulator removed; skipped unless REVOLVE_LEGACY=1")


def pytest_collection_modifyitems(config, items):
    """Tests marked `gpu` are skipped unless EVAL_GPU=1 (the unmarked ones, this dir's older GPU tests included, run as before)."""
    legacy.mark(items, LEGACY)
    if os.environ.get("EVAL_GPU") != "1":
        skip = pytest.mark.skip(reason="needs a GPU: set EVAL_GPU=1")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
