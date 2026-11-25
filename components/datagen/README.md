# datagen

Datagen makes demonstration episodes. It samples scene instances (the scene with a randomised robot start), runs
each one on its own simulator with a scripted expert, and keeps every episode, passed or failed. Only the
simulator's evaluator decides whether an episode passed. Its output is the input of the dataset stage.

## Inputs

| Input | Comes from | Format |
|---|---|---|
| Datagen spec | `specs/datagen/*.yaml` (`smoke_v2`, `fridge_panda_v2`, `drawer_so101xl_v2`) | Required: `task`, `robot`, `out` (under `runs/`), `generator`, `splits`. Optional: `randomisers`, `validators`, `checks`, `seed`, `scene`, `runner`, `converter`, `run` |
| Or a job file | another tool that already made the scene instances | `task`, `robot`, `out`, `episodes: [{seed, scene}]`, `episodes_per_scene` (DATAGEN-v2.md 8.2) |
| Task, robot, scene specs | `specs/tasks/`, `specs/robots/`, `specs/scenes/` | YAML, checked by `./revolve validate` |
| Simulator | `./revolve build simulator` plus `assets/` | needed by `generate` only, with a GPU |
| Command line | you | step name, `--split S`, `--workers N` (sims at a time), `--gui` or `--headless`, `--dry-run` |
| Extra plugins | `DATAGEN_PLUGIN_PATH` | directories with plugin modules |

Generators: `moveit` (a scripted MoveIt expert for `open_fridge_door` and `open_drawer`, run in a client container
from the simulator image) and `automoma` (an AutoMoMa and cuRobo whole-body plan per scene instance). `automoma`
needs the `revolve-datagen` image, which nothing on main builds today, so only `moveit` can run. The shipped specs
all use `moveit`.

The only randomiser is `robot_start`: base `x_m`, `y_m`, `yaw_rad` (or the offsets `dx_m`, `dy_m`, `dyaw_rad`)
and the `arm` pose. `--list-randomisers` prints their ranges for a task and robot.

## Outputs

| Output | Where it lands | Read by |
|---|---|---|
| Scene instance | `<out>/<split>/scene_<seed>/scene.yaml` | the simulator (`--scene`) |
| Episode results and bags | `<out>/<split>/scene_<seed>/episode_<n>.json`, `episode_<n>_bag/`, `episode_<n>_gz.log` (written by the simulator) | dataset |
| Try log per scene | `<out>/<split>/scene_<seed>/datagen.json` | datagen, when it resumes |
| Manifest | `<out>/manifest.json`: one entry per episode attempted, with `passed` copied from the result | dataset |
| Run state and report | `<out>/state.json`, `<out>/report.md` | datagen (resume), you |
| Logs | `<out>/logs/scene_<seed>.sim.log`, `.client.log` | you, when debugging |

A job file runs as one split named `job`.

## Run

```
./revolve datagen specs/datagen/smoke_v2.yaml                 # all steps: sample, generate, convert, check
./revolve datagen specs/datagen/smoke_v2.yaml sample          # sample the scene instances only (no Docker)
./revolve datagen specs/datagen/smoke_v2.yaml all --dry-run   # say what would run, write nothing
./revolve datagen --list-randomisers --task open_drawer --robot mobile_so101_xl
python components/datagen/mock_manager.py --workers 2 specs/datagen/smoke_v2.yaml   # stand-in for the manager
```

A rerun resumes from `<out>/state.json`. Exit codes: 0 on success, 1 for a failed check or an error,
128 + n when stopped by signal n (the state is saved and the run's sims are stopped).

The `convert` step is skipped today: its log says the new bags have no Rosetta contract yet. The robots now have
contracts (`specs/robots/<robot>/contract.yaml`), but turning bags into a training dataset is still the dataset
component's work.

## Test

```
./revolve test datagen                  # CPU only, fake plugins, no Docker: about 20 s
DATAGEN_GPU=1 ./revolve test datagen    # adds the GPU tests (marker `gpu`), about 8 min
```

The GPU tests start the simulator image on the local GPU: a sim session, the MoveIt expert for each task, the
acceptance run of `mock_manager.py` on `smoke_v2`, the output contract and the AutoMoMa plan. The AutoMoMa test
also needs the `revolve-datagen` image. They remove their containers and their `runs/*_gpu_*` outputs.

## Files

- `src/datagen/cli.py`: the `./revolve datagen` command.
- `src/datagen/core/pipeline.py`: the four steps and the output layout.
- `src/datagen/core/spec.py`: the spec format, with an error for every bad key.
- `src/datagen/plugins/`: generators, randomisers, validators, runners and the converter.
- `mock_manager.py`: hands datagen a spec or job file, as the manager will.

The contract with the simulator, the job file and the manifest: [DATAGEN-v2.md](DATAGEN-v2.md).
