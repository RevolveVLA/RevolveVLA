# evaluation
Policy rollouts on the simulator for a datagen training spec: held-out scene instances (new) and training instances (seen),
each judged by the simulator's evaluator (judge `c0` = its `passed`). Reuses datagen's samplers, validators, instancer and pool.

## Run
```
./revolve evaluate specs/eval/smoke_v2.yaml [sample|check|rollout|report|all] [--set S] [--workers N] [--gui|--headless] [--dry-run]
```
`rollout` needs the simulator image, the assets and a GPU. A re-run resumes from `<out>/results.json`; a changed spec refuses an
existing `out`. Exit codes: 0 ok, 1 failed check or error, 2 invalid spec, 128 + n when stopped by signal n (the run's containers
are removed).

## Spec (`specs/eval/*.yaml`)
Required: `train` (the training datagen spec), `out` (inside runs/), `sets` (`heldout: {count, sampler, pool, min_distance}`,
`train: {from: train, count}`). Optional: `training` (`datagen`, default: reads the training run's `state.json` + `manifest.json`),
`policy` (`oracle` replays a training bag's actions, `expert` runs the training spec's generator closed loop, `smolvla` serves a
checkpoint: blocked until the new bags have a Rosetta contract), `ckpt` (`local` path or `hf` repo[@revision][/subfolder]),
`judge: {criterion: c0}`, `rollout` (`sim`, or `fake` for tests; params `record`, `wall_s`, `ready_s`, `infra_retries`, `lead_s`),
`rollouts` per config, `retries` (new episode from the start), `runner` (policy runner chunking), `run: {workers, headless}`.
Plugins live under `src/evaluation/plugins/<kinds>/`.

## Outputs (`<out>/`)
`<set>/state.json` + `<set>/scene_<seed>/scene.yaml` (the instances), `<set>/<seed>_sim/` (the config's sim: episode results and bags), `<set>/<seed>_r<i>/` (+ `retry_<j>/`:
`episode.json`, `rollout.json`, client log), `results.json` (every try: passed, phases, termination, judge detail), `report.md`.

## Tests
```
./revolve test evaluation               # CPU: unit + system tests with fake plugins
EVAL_GPU=1 ./revolve test evaluation -k gpu   # the expert policy on the real sim (needs the simulator image, assets, a GPU)
```
