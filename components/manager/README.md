# manager

The manager runs the experiment loop: datagen, then dataset, training and evaluation, then recovery on the
failures, turn after turn. It runs each step in its own container, watches it, retries it once, and can resume
after a crash.

Status on main: the manager does not run. Its `./revolve run`, `resume`, `status` and `stop` commands were removed
from the CLI together with the experiment specs (`specs/experiments/`). Its code still calls
`specs.load_experiment` and the step scripts `generate.py`, `starts.py` and `evaluate.py`, which no longer exist.
The rest of this page describes the design, for when the manager is ported to the new components.

## Inputs

| Input | Comes from | Format |
|---|---|---|
| Experiment spec | `specs/experiments/<exp>.yaml` (removed) | task, robot, seeds (train, probe, held-out), training block, `max_turns`, `stall_timeout_s` |
| Component entry points | each component | datagen, evaluation and recovery CLIs; dataset `launch.turn_dataset`; training `launch.start_train` |
| Docker | the host's Docker socket | the manager and every step run as containers from `revolve-manager` |
| Environment | your shell | every `REVOLVE_*` variable is passed on to the steps; `REVOLVE_GPU_PREFLIGHT=0` skips the GPU checks |

## Outputs

| Output | Where it lands | Read by |
|---|---|---|
| Loop state | `runs/<exp>/state.json`, written before each step starts | the manager, when it resumes |
| Summary | `runs/<exp>/summary.json`: per turn the step states, held-out and probe results, dataset size, final loss; the stop reason | you |
| Turn data | `runs/<exp>/turn_<k>/`: `data/` or `recovery/`, `dataset/`, `ckpt/`, `eval/heldout/`, `eval/probe/` | the next step |
| Step records | `runs/<exp>/turn_<k>/steps/<step>_a<n>/`: `job.json`, `status.json`, `result.json`, `stdout.log`, `container.log` | you |
| Log | `runs/<exp>/manager.log` | you |

The loop stops when every held-out seed passes every phase on the first try, or after `max_turns`. Episodes from
held-out seeds never reach training: a leak stops the experiment with `heldout_leak`.

## Run

```
./revolve build manager       # the revolve-manager image: python:3.12-slim plus requirements.txt
```

Nothing else runs today. The removed commands were `./revolve run <exp>`, `status <exp>`, `resume <exp>` and
`stop <exp>`.

## Test

```
./revolve test manager        # unit tests (tests/) and the stub loop (tests/integration/)
```

On main the unit tests fail: 9 failed and 3 errors out of 13, on the missing `specs.load_experiment` and
`specs/experiments/fridge_mini.yaml`. The stub loop needs the `revolve-manager` image and the same missing pieces.

## Files

- `src/manager/plan.py`: the loop as data (which step is next, its inputs, the stop rule), no Docker.
- `src/manager/loop.py`: runs the steps in containers, polls their `/status`, handles stalls, retries and resume.
- `src/manager/steps.py`: one step inside its container.
- `src/manager/cli.py`: the host side of the removed `run`, `resume`, `status` and `stop` commands.
