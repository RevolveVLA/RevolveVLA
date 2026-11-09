# recovery
Recovery demonstrations from a failed rollout's saved phase state: the failed phase and the rest of the task
re-planned with AutoMoMa + cuRobo from that state, validated in the simulator from the restored state, recorded
as suffix-only episodes labelled recovery + phase. Ported from the earlier pipeline's regenerate stage
(regen, plan_from_state.py, simulator 
regenerate.py + gz_regen.sh); the algorithm, budget and thresholds are unchanged.

## Build and test
```
./revolve build recovery    # revolve-recovery FROM revolve-datagen (builds datagen first if it is missing)
./revolve test recovery     # unit tests (stubbed jobs) + real recoveries for the fridge and the drawer
```
- `Dockerfile`: `FROM ${DATAGEN_IMAGE}` (default revolve-datagen:latest) + a label; nothing is added (same
  layers). cuRobo and AutoMoMa are mounted as for datagen.
- Also needs `./revolve build simulator` (the gz side runs in revolve-simulator) and the kitchen assets in
  assets/ (components/simulator/README.md).
- Test fixtures are generated, not committed: a datagen episode per task and a perturbed fridge open start,
  cached in runs/test_cache/recovery/ (delete it to regenerate). First run ~11 min, later runs shorter.

## Entrypoints
| What | Where | Runs in |
|---|---|---|
| `recovery_entry.py plan --dir D --seed S --attempts N` | `src/recovery/` | revolve-recovery, GPU, under `revolve_common.wrapper --parser recovery_progress:parse` |
| `regen_sim.py candidate --plan P --attempt A --out D [--cameras HZ]` | `src/recovery/` | revolve-simulator (the simulator's launcher with `cli=`) |
| `regen_sim.py perturb --source EP --out D --joint X` | `src/recovery/` | revolve-simulator; test support (a policy-like open start) |
| `recover.py --task T --rollouts R1[:PHASE] ... --out runs/<exp>/turn_<k>/recovery [--episodes-per-failure 3] [--phases approach,grasp,open] [--workers 2]` / `recover(...)` | `recover.py` | host or the manager's container (docker SDK); PROGRESS lines for a wrapper |

`recover` = per rollout, per episode i (of episodes_per_failure): planner runs with seeds seed0 + runs*i + r
(generator.yaml `regenerate:` runs 3, attempts 10, seed0 100), each planned attempt a camera-off candidate run
(regenerated actions -> clearance precheck -> gz restored from the saved state), the first accepted one recorded
with the cameras (same state_log.npz required, bag check) = episode i. Plan jobs share a process-wide GPU cap
(REVOLVE_GPU_PLANNERS, default 2: 3 OOM on the 8 GB laptop).

| Failed phase | Plan (mode) | Restored from | Judged from |
|---|---|---|---|
| approach | MotionGen reach to a new grasp + its AKR pull (reach) | approach_start.npz | approach |
| grasp | the same (reach) | grasp_start.npz | approach (it re-approaches) |
| open | AKR pull from the held state (pull); a policy rollout's on the grasp it holds (grasp_from_state) | open_start.npz | open |
| release, global (untimed) | not regenerated: `unsupported` (legacy has none) | - | - |

Accepted = every task.yaml phase from the judged one on passes AND the offline verdict of the regenerated part
(validation.py, the original thresholds) is accepted (the simulator's live_validate.finalize in replay mode).

## Inputs: a rollout dir
`phase_states/<phase>_start.npz` + `_start_next.npz` (the simulator's phase states), `phase_report.json`
(`result.failed_phase`, `result.regenerate`), `episode.json` (`task`; optional `seed`, `policy_rollout`,
`plan_run`, `attempt_id`, `grasp_id`). An open restart plans on the held grasp when `policy_rollout` is present
(or live.json `policy`) or the source grasp id is unknown. `R:PHASE` forces a phase (an accepted demo's states).

## Outputs (`<out>/`)
`manifest.json` (rollouts -> source, state done|short|unsupported|skipped|passed|error, plans, checks,
episodes; `episodes` = accepted dirs; `counts`), and per rollout `<name>/` (`<rollout>` or
`<rollout>_from_<phase>` when forced): `plans/e<i>_r<r>_s<seed>/`, `checks/<plan>_a<a>/`,
`episodes/ep_<i>/` (bag/, bag_check.json, verdict.json, phase_report.json, phase_states/, state_log.npz,
episode.json with `regenerated` and `recovery` = {source: recovery, label: recovery, phase, suffix: true,
from_rollout, rollout_seed, mode, grasp_from_state, action_index_offset, plan, planner_seed, ...}),
`jobs/<kind>_<name>/`. A re-run resumes from the manifest (a crashed plan run is redone).
