# datagen
AutoMoMa + cuRobo expert demonstrations: per seed a plan run, then the planned attempts replay-checked and
recorded in the simulator until the seed has its accepted episodes. Ported from
the earlier pipeline's AutoMoMa planner (unchanged algorithm) and its stages (plan -> check -> record,
manifest resume). Task, robot and scene come from specs/ (revolve_common); the planner's robot code and SDF
helpers are the simulator component's copies (components/simulator/src/sim).

## Build and test
```
./revolve build datagen [--clean]   # image revolve-datagen (deps only), then cuRobo built inside it
./revolve test datagen              # unit tests (stubbed jobs) + real jobs for seed 0 of fridge and drawer
```
- `Dockerfile`: dependencies only (CUDA 12.8 devel, uv Python 3.11, torch 2.7.0 cu128, the plan-step pip deps
  as the legacy planner image, + pydantic/FastAPI/uvicorn). GPU arch: build arg `TORCH_CUDA_ARCH_LIST`
  (`REVOLVE_GPU_ARCH`, default 12.0 = sm_120; 8.9 for Modal L4).
- `build.sh`: cuRobo's CUDA extensions compiled inside the image from a copy of ThirdParty/curobo (+
  ThirdParty/patches/curobo/*.patch) into `ThirdParty/curobo/install` (git-ignored, `ignore = untracked` in
  .gitmodules), mounted read-only at /opt/curobo/install. AutoMoMa is pure Python: ThirdParty/automoma is
  mounted read-only at /opt/automoma and put on PYTHONPATH (what its editable install did).
- Also needs `./revolve build simulator` (the replays) and the kitchen assets in assets/ (components/simulator/README.md).

## Entrypoints
| What | Where | Runs in |
|---|---|---|
| `entry.py plan --task T --seed S --out D [--attempts N]` | `src/datagen/entry.py` | revolve-datagen, GPU, under `revolve_common.wrapper --parser progress:parse` (`launch.py`) |
| `generate.py --task T --seeds 0-9 --out runs/<exp>/turn_0/data [--episodes-per-seed 1] [--workers 2]` / `generate(...)` | `generate.py` | host or the manager's container (docker SDK); may itself run under the wrapper (PROGRESS lines) |
`generate` = per seed: the datagen `plan` job, then simulator `replay` jobs (components/simulator/launch.py):
a camera-off check per planned attempt and, when it is accepted, the camera replay that records the bag.
`workers` seeds run at once (= planners on the GPU). Measured on the 8 GB laptop: 2 planners peak at 4.9 GB
(default 2).
An episode is accepted when the check and the record both pass (replay exit 0: offline verdict accepted and
every task.yaml phase passed), the record reproduces the check's state_log.npz (sha256), and its bag check passes.

## Inputs
specs/tasks/<t>/task.yaml (-> robot, scene) through `src/datagen/specs_bridge.py` (the old plugin dicts the
planner reads). Generator settings: `src/datagen/generator.yaml` (the old generator plugin.yaml: attempts 50,
cuRobo settings) + per task the v2 rounds' `starts` (clustered starts; plan job ROUND_STARTS) and
`timing`/`pull` blocks (replay ROUND_TIMING / ROUND_PULL). The starts' centre seed is the seed itself.

## Outputs (`<out>/`)
`manifest.json` (seeds -> plan, checks, episodes, state done|short|failed|error; `episodes` = accepted names;
`counts`), `plans/seed_<s>/` (handoff.npz/json, plan_report.json, fk_check.json, assets/),
`checks/seed_<s>_a<a>/`, `episodes/seed_<s>_a<a>/` (bag/, bag_check.json, verdict.json, phase_report.json,
phase_states/, state_log.npz, episode.json with a `datagen` block, plan.json + plan.npz), `jobs/<kind>_<name>/`
(job.json, status.json, result.json, stdout.log, container.log). A re-run resumes from the manifest.
