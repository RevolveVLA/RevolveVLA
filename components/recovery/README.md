# recovery

Phase-wise recovery of a finished evaluation run: the failed phases of its rollouts are found in their bags, restarted from the
robot state shortly before the failure and recorded again by a datagen generator, up to a quota of episodes per phase.
Same shape as datagen and evaluation: `core/` + `plugins/`, one yaml, its own entrypoint.

## Run
```
./revolve recover specs/recovery/fridge.yaml [collect|run|check|live|all] [--workers N] [--gui|--headless] [--dry-run]
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
- `live`: a live recorder plugin that runs inside rollouts (`core/interfaces.py`): `dagger` supervises the policy with privileged state
  (`triggers`, as in `revolve_common.supervision`), checks before each action, and at the first firing an expert (`moveit_takeover`: the
  moveit expert's `resume`) takes over for good: the policy stops publishing and the rest of its chunk is dropped. With `from: {eval:
  <evaluation spec>, sets: [..], heldout: [..]}` the stage `live` rolls that spec's sets out with its policy (the evaluation's own
  pipeline and `sim` rollout, in `<out>/live/`) under the recorder; `from.sets` default = its sets that are not held-out, `from.heldout`
  = the sets that must stay unseen (default: its held-out sets); seeds that overlap them, or the training spec's aside pools, are refused.
  Params: `triggers` (required), `expert`, `record: full | interventions`, `quota {phase: takeovers}` (a full phase gets no more
  takeovers: the evaluator's verdict stands; checked when an episode starts, so parallel workers can overshoot by a few). Policies that
  can be halted stream actions (`oracle`, `frozen`, `noisy_expert`, a served policy); `expert` cannot. A spec has `phases`, `live` or both
  (phase-wise recovery needs `from.eval` = a finished run dir, a live run an evaluation spec: two specs then).

## Outputs (`<out>/`)
`state.json`, `manifest.json` (datagen's episode list with a `recovery` block per entry; `core/manifest.py`), `report.md` (per phase:
failures, unreproducible, selected, recorded, quota fill), `<phase>/<failure id>/{scene.yaml, datagen.json, resets.json}` (a job),
`sims/<slot>/` (the sims' episode results and bags), `logs/`; a live run: `live/` (an evaluation run: results.json, report.md,
`<set>/<seed>_sim/episode_<n>.{json,_bag,interventions.json}`, `summary.json`). The sidecar `episode_<n>.interventions.json`
(`core/flags.py`) holds the flags a bag cannot: `takeover_t`, `trigger`, `phase`, `segments [[t0, t1, "policy" | "expert"]]`, ...; the
manifest has one `recovery.kind: dagger` entry per episode with a takeover.

## Plugins (`src/recovery/plugins/<kinds>/`, or modules on $RECOVERY_PLUGIN_PATH)
`runner` (`sim`, `fake`), `scanner` (`container`: the scan in the simulator image; `local`: in this process), `live` (`dagger`), `expert`
(`moveit_takeover`); `policies/` (`frozen`, `noisy_expert`: bad policies for the evaluation registry that force the triggers).

## Tests
```
./revolve test recovery                       # CPU: unit + system tests with fakes (fake eval run + bags, fake sim)
```
