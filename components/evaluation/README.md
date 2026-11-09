# evaluation
Closed-loop policy rollouts in the simulator: per seed and per phase verdicts from task.yaml, judged live by the
simulator's phase validator, with every phase start state saved. Ported from the earlier pipeline's evaluate.py,
rosetta_goal.py and the gz_kura_mujoco closed loop (closed_loop_system.py, ros/ros_relay.py,
ros/dummy_policy.py); the legacy breakage is not copied (no evals/*.yaml, no closed_loop_success block, no
subprocess PYTHONPATH reset, no fixed 285 s wall limit).

No image of its own (evaluation is part of the training env). A rollout runs, on its own
`--internal` docker network with its own `ROS_DOMAIN_ID` (the manager never joins ROS):

| Container | Image | What |
|---|---|---|
| serve | revolve-training | `entry.py serve` (Rosetta's `lerobot_rosetta.policy_server`, GPU), `components/training/launch.py start_serve` |
| client | revolve-dataset + ThirdParty/rosetta/install | Rosetta's own `policy_runner_node` (`ros2 launch rosetta_ros policy_runner_launch.py`, `use_sim_time`) + `src/evaluation/rosetta_goal.py` (one RunPolicy goal); the oracle: `src/evaluation/dummy_policy.py` |
| sim | revolve-simulator-ros | `components/simulator/src/sim/cli.py rollout`: `closed_loop_system` (rebased on the live phase validator, policy mode) + `ros_relay.py` (ROS 2 <-> gz-transport) |

revolve-simulator-ros is the thin deps-only ROS 2 Jazzy layer FROM revolve-simulator
(`components/simulator/ros/Dockerfile`, the legacy ros/Dockerfile); ROS is sourced for the relay process only.

## Build and test
```
./revolve build evaluation    # revolve-simulator-ros (+ simulator / training / dataset first if missing)
./revolve test evaluation     # unit tests (stubs) + real rollouts (tests/conftest.py)
```

## Entrypoints
| What | Where |
|---|---|
| `evaluate(task, ckpt, seeds, out, set_name, starts_dir, policy, rollouts_per_seed, retries, parallel, sources, budget_s, wait_s, gui)` / `evaluate.py --task T --ckpt C --seeds 100-103 --set heldout --out runs/<exp>/turn_<k>/eval --starts runs/<exp>/starts` | `evaluate.py` (host or the manager's container; under `revolve_common.wrapper --parser eval_progress:parse`: PROGRESS rollouts / rollouts_done / passed / first_try_passed / running) |
| `run(start, out, seed, policy, ckpt, restore, budget_s, wait_s, rtf, gui)` one rollout | `rollout.py` |
| `resolve(task, seeds, out, sources)` seed -> start / `starts.py --task T --seeds S --out D` | `starts.py` |

## Watching a rollout (`--gui`)
Opt-in; the default and the tests stay headless. The sim container also starts a `gz sim -g` client on the host's
X display (GNOME/Wayland: Xwayland), attached to the rollout's server; `--gui` forces `--parallel 1`:

```bash
.venv/bin/python components/evaluation/evaluate.py --task open_fridge_door --ckpt runs/fridge_mini/turn_2/ckpt \
    --seeds 100 --set heldout --starts runs/fridge_mini/starts --out runs/infer_demo/eval --gui
```

`--policy oracle` (no `--ckpt`) replays the start's expert actions, so the door visibly opens. Needs `DISPLAY` and
`XAUTHORITY` in the shell (a desktop terminal has both): the container gets DISPLAY, `/tmp/.X11-unix` and the
XAUTHORITY file (read-only), runs as your uid and renders with the NVIDIA GPU; no `xhost` change. The rollout's
network stays `--internal`. The GUI is not part of the verdict: closing its window leaves the server running and
the rollout finishes as headless; the window closes when the rollout ends. Its log: `<seed>_r<i>/gz_gui.log`. A
finished rollout dir is kept (resume): use a fresh `--out` to watch the same seed again.

## Seed -> start
The first replay-accepted attempt of the seed's datagen plan run (AutoMoMa planner seed = the seed, clustered
starts centred by it): a train seed's start is its demo's, a probe / held-out seed's is what a demo of it would
start from (the legacy eval round `--until mint`). Taken from a datagen output dir when it has one (`--source`),
else planned + checked into `<starts>/plans|checks/`. Cached in `<starts>/seed_<S>/start.json` (sha-checked): the
same seed gives the same start for the whole experiment.

## Time, retries, parallelism
- Sim time from task.yaml `rollout`: wait_s before the first action, budget_s after it, after_success_s run-on
  (the closed loop's hold check ends the run early). Wall-clock limit = that sim time / the measured RTF x 2
  (`rollout.wall_limit`; RTF: the lowest measured in this eval, else `runs/eval_rtf.json` per parallel count, else
  0.1) + 240 s setup.
- A phase still running at the end fails (`phase_ended`), so recovery restarts the phase the policy was in.
- `retries` (experiment eval.retries, default 0): the policy runs again from the failed try's
  `phase_states/<phase>_start.npz` (restored in step 0, spawn target = its commanded action) in
  `<seed>_r<i>/retry_<j>/`. A stack error (no verdict) is re-run once. A global (untimed) failure has no phase state:
  it is not retried.
- `parallel` default: free GPU memory / 1.7 GB per SmolVLA rollout (measured), at most 4.

## Outputs (`<out>/<set>/`)
`<seed>_r<i>/` = the first try = recovery's input (components/recovery/recover.py `source()`):
phase_report.json (`result.failed_phase`, `regenerate`), phase_states/<phase>_start(_next).npz, episode.json
(task, seed, plan_run, attempt_id, grasp_id, policy_rollout), live.json (policy mode); plus closed_loop.json,
state_log.npz, actions.npz, summary.json, relay.json, client/ (runner logs, runner.json / dummy_policy.json),
jobs/sim, jobs/serve, rollout.json. `eval_summary.json`: seed x phase table (first try, retries), first-try
success, all_pass (every seed every phase, first try), all_pass_with_retries, RTF, wall times, GPU memory.
