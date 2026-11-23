# datagen
Scene-instance demonstrations on the simulator (contract: [DATAGEN-v2.md](DATAGEN-v2.md)). Per split the pipeline samples
scene instances (robot start, scene), runs each one on the sim (`./revolve sim`, one sim + one client container per
instance) with an expert, and keeps every episode the simulator's evaluator accepts. The sim is the only recorder.

## Run
```
./revolve datagen specs/datagen/smoke_v2.yaml [sample|generate|convert|check|all] [--split S] [--workers N] [--gui] [--dry-run]
./revolve datagen --list-randomisers --task T --robot R      # what a task + robot can vary
python components/datagen/mock_manager.py [--workers N] <spec.yaml | job.yaml>   # stand-in for the manager (DATAGEN-v2.md 8.2)
```
`sample` needs no docker. `generate` needs the simulator image (`./revolve build simulator`), the scene assets and a GPU.
A re-run resumes from `<out>/state.json`. Exit codes: 0 ok, 1 failed check or error, 128 + n when stopped by signal n.

## Spec (`specs/datagen/*.yaml`)
Required: `task`, `robot`, `out` (inside runs/), `generator` (`plugin: moveit | automoma`, `attempts`, `episodes_per_scene`, `record:
images|no-images`), `splits` (`<name>: {target, sampler, pool: {initial, replacement}, min_distance, record}`).
Optional: `randomisers` (`robot_start`: `x_m`, `y_m`, `yaw_rad`, `arm`, as ranges or choices), `validators` (`collision_free`),
`checks`, `seed`, `scene`, `runner` (`sim`, or `fake` for tests), `converter` (`rosetta`: bags -> LeRobot through components/dataset; skipped until the new bags have a Rosetta
contract), `run: {workers, headless}`. `automoma` (open_fridge_door, mobile_panda) plans each instance first in the `revolve-datagen` image (cuRobo built
in ThirdParty/curobo/install, GPU; one plan at a time, `<scene>/plan.json`) and streams the plan to the sim. Plugins live under `src/datagen/plugins/<kinds>/` and register with the core registry.

## Outputs (`<out>/`)
`<split>/scene_<seed>/` (`scene.yaml`, `datagen.json` the log of the tries, `episode_<n>.json` the evaluator result,
`episode_<n>_bag/` the sim's bag), `manifest.json` (DATAGEN-v2.md 8.4: one entry per attempt, accepted or not), `state.json`
(resumable state), `report.md`.

## Tests
```
./revolve test datagen                  # CPU: unit + system tests with fake plugins
DATAGEN_GPU=1 ./revolve test datagen    # + GPU tests (marker `gpu`): sim session, moveit expert per task, automoma plan + execute, acceptance, bag contract
```
The GPU tests start the simulator image on the local GPU (about 8 min) and remove their containers and `runs/*_gpu_*` outputs;
`-k "gpu or acceptance or contract"` selects only those.
