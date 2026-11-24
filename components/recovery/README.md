# recovery

Phase-wise recovery of a finished evaluation run: the failed phases of its rollouts are found in their bags, restarted from the
robot state shortly before the failure and recorded again by a datagen generator, up to a quota of episodes per phase.
Same shape as datagen and evaluation: `core/` + `plugins/`, one yaml, its own entrypoint.

## Run
```
./revolve recover specs/recovery/fridge.yaml [collect|run|check|all] [--workers N] [--gui|--headless] [--dry-run]
```
`collect` and `run` need the simulator image (the bags are read with rosbag2_py), the assets and a GPU. A re-run resumes from
`<out>/state.json` (the collected failures, the selection, the finished jobs); a changed spec refuses an existing `out`. Exit codes:
0 ok, 1 failed check or error, 2 invalid spec, 128 + n when stopped by signal n (the run's containers are removed).

## How it works
- `collect`: every failed try of the eval run (`<from.eval>/results.json`) goes through `triggers` over its recorded bag (default: the
  evaluator's verdict; a `stall` or `condition` trigger can name an earlier failure point or another phase). The restart is the
  failed phase's start (`restart_from: phase_start`) or `back_s` before the failure, with the robot's base pose and arm joints from
  the bag. A state in which a scene joint has moved (the door is open) cannot be reproduced (a start cannot set joints): such failures
  are counted as `unreproducible`, with the value, and not recovered.
- `run`: per phase `quota` episodes are wanted (`per_failure` accepted episodes per failure): the failures are chosen round robin over
  the eval configs and interleaved over the phases (seeded), each becomes a job: the failure's scene with `robot_start` = the restart,
  recorded by `phases.generator` (a datagen generator) in up to `attempts` tries. `run.workers` jobs run at a time, each on a sim slot.
  `run.reset: warm` reuses a running sim of the same world (reset, then drive to the restart); a miss relaunches the sim at the
  restart. `relaunch` launches a sim per job.
- `check`: every job ran and every phase has its quota of recorded episodes. `keep: successes` records only the tries the evaluator accepted.
- `live`: an optional live recorder plugin (it runs inside rollouts; `core/interfaces.py` has its interface). Here it is only validated.

## Outputs (`<out>/`)
`state.json`, `manifest.json` (datagen's episode list with a `recovery` block per entry; `core/manifest.py`), `report.md` (per phase:
failures, unreproducible, selected, recorded, quota fill), `<phase>/<failure id>/{scene.yaml, datagen.json, resets.json}` (a job),
`sims/<slot>/` (the sims' episode results and bags), `logs/`.

## Plugins (`src/recovery/plugins/<kinds>/`, or modules on $RECOVERY_PLUGIN_PATH)
`runner` (`sim`, `fake`), `scanner` (`container`: the scan in the simulator image; `local`: in this process), `live`, `expert`.

## Tests
```
./revolve test recovery                       # CPU: unit + system tests with fakes (fake eval run + bags, fake sim)
```
