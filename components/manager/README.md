# manager
The experiment loop: per turn, generate (turn 0) or recover (later turns) -> dataset -> train -> eval, until
every held-out seed passes every phase on the first try, or max_turns.

```
./revolve build manager                     revolve-manager: python:3.12-slim + requirements.txt (deps only)
./revolve run <exp | specs/.../exp.yaml>    starts the manager container (detached); log runs/<exp>/manager.log
./revolve status <exp>                      seed x phase table per turn + the running step's live progress
./revolve resume <exp>                      restarts the manager on runs/<exp>/state.json
./revolve stop <exp>                        kills the manager, cancels the running step and what it started
./revolve test manager                      unit tests (tests/) + the stub loop (tests/integration/), ~3 min
```

## Loop (src/manager/plan.py)
| turn | steps |
|---|---|
| 0 | generate (train seeds) -> starts (held-out + probe, once per experiment, runs/<exp>/starts) -> dataset -> train (from training.base) -> eval_heldout -> eval_probe |
| k > 0 | recover (turn k-1's probe rollouts that failed a phase on the first try) -> dataset (turn k's episodes merged after turn k-1's) -> train (training.mode, prev = turn k-1's ckpt) -> eval_heldout -> eval_probe |

- Stop: turn k's held-out `eval_summary.json` `all_pass` (first try; retries only add diagnostics), or k + 1 == max_turns.
  eval_probe is skipped when no turn follows. A turn with no probe failure / no recovered episode trains on the last dataset.
- Held-out data never reaches training: every episode handed to the dataset step is checked (its seed is not held-out,
  it is not under an eval/heldout dir) and recovery inputs must be this turn's probe rollouts; a violation stops the
  experiment (`heldout_leak`).
- Steps call the components' own composition entry points: generate.py, starts.py, recover.py, evaluate.py (their
  CLIs), dataset `launch.turn_dataset`, training `launch.start_train` (backend + mode from the yaml).

## Containers (src/manager/loop.py, steps.py)
- The manager and every step run in revolve-manager with the repo mounted at its **host path** (rw) and the docker
  socket, as the host uid + the socket's group, /etc/passwd + /etc/group read-only. The components' launchers compute
  their bind mounts from their own file paths, so these are valid host paths for the sibling containers they start.
- A step = one container `revolve-<exp>-t<k>-<step>-a<n>` running revolve_common.wrapper -> `python -m manager.steps`.
  Every container / network a step creates is labelled `revolve.exp=<exp>`, `revolve.parent=<step container>`
  (a docker SDK create() hook in steps.py), so a cancelled step's children are removed.
- The manager polls `http://<step IP>:8000/status` on docker's default bridge (no published ports, as every component
  job), falls back to status.json. Training's step / loss are relayed into the step's /status.
- Evaluation steps get the GPU with NVIDIA_DRIVER_CAPABILITIES=utility (nvidia-smi only: evaluate sizes its parallelism
  from free GPU memory); the GPU work itself runs in the component containers.

## Failures, stalls, resume
- Stall: no change in the step's progress / last line / newest sub-job status.json under its output dir for
  stall_timeout_s -> POST /cancel, stop, children removed; attempt "stalled".
- A failed or stalled step is retried once (the components resume from their manifests); a second failure stops the
  experiment: status failed, `error` in summary.json. A step container docker cannot start is removed (failed attempt).
- GPU preflight (revolve_common/gpu.py): before generate / starts / recover / eval_* (and train on the local backend)
  and every REVOLVE_GPU_CHECK_S (60 s) while one runs, nvidia-smi in a throwaway manager-image container (utility caps
  only). A broken GPU (exit code, `ERR!`, "requires reset", or docker cannot start a GPU container) stops the
  experiment at once: stop_reason gpu_unhealthy, the step back to pending (not a failure), so `./revolve resume`
  carries on after a reboot. The GPU launchers (simulator, datagen, training) check host nvidia-smi the same way when
  they have one. REVOLVE_GPU_PREFLIGHT=0 turns it off.
- runs/<exp>/state.json (atomic) is written before each step container starts. A restarted manager re-attaches to a
  running step container, or reads the result.json of one that finished while it was down; done steps never rerun.
  The experiment yaml must not change between run and resume (sha256 checked); after a deliberate edit,
  `./revolve resume <exp> --accept-spec-change` records the new sha256 (old one in state.json spec_changes) and
  carries on: done steps are kept, the pending ones use the edited yaml.

## Outputs
runs/<exp>/: state.json, summary.json (per turn: step states / attempts / wall, held-out and probe seed x phase tables,
dataset size, training steps + final loss; stop reason, error), manager.log, starts/, turn_<k>/{data | recovery,
dataset, ckpt, eval/{heldout,probe}, jobs, steps/<step>_a<n>/ (job.json, status.json, result.json, stdout.log,
container.log)}.
