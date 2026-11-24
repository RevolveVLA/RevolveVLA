# recipe

Picks the data of the next training round from finished runs: how many episodes (or frames) of each kind (base datagen, evaluation
rollouts, phase-wise recovery, DAgger) and phase, divided into train / val by scene, minus the held-out scenes. One yaml; same shape
as datagen and evaluation (`core/` + `plugins/`). It reads files only: no docker, ROS, GPU or sim.

## Run
```
./revolve recipe specs/recipes/fridge_round1.yaml [resolve|check|materialise] [--dry-run]
```
`resolve`: what each source holds (usable items per phase, refused, left out). `check`: the plan (asked vs got per source x phase x split,
the guard, the invariants) on stdout, nothing written. `materialise` (default): check, then `<out>/report.md` and `<out>/selection.json`.
Exit codes: 0 ok, 1 `on_short: error` and short / a broken invariant / an unreadable run, 2 invalid yaml.

## Recipe yaml
`sources` (a plugin each), `take` (an amount per source), `splits`, `exclude`, `unit`, `fps`, `on_short`, `seed`, `out`, `output`
(default `selection`). See `specs/recipes/fridge_round1.yaml`.
- `take`: `all` | n | `{phase: n | all}`, in the unit (`episodes`; `frames` = the seconds of the segment or the episode x `fps`, taken until the
  amount is reached or passed). Balanced over scenes (one item of each in turn), shuffled by `seed`: a few scenes cannot fill a quota.
- `splits`: fractions of the unit, grouped by scene: every item that starts from a scene (any source) is in one split.
- `exclude`: folders (a held-out set's: its `scene.yaml` files name the scenes, `scene_<seed>` the seeds) or scene files. Refused: an item
  whose scene or episode folder is under one, or whose config seed is one of the excluded seeds. An exclude with no scene file is an error.

## Sources (`plugins/sources/`)
- `datagen`: `manifest` (datagen's manifest.json), `passed` (true default | false | any).
- `rollouts`: `run` (an evaluation out: `results.json`), `set`, `passed`. One item per try with a verdict and a bag.
- `recovery`: `run` (a recovery out: `manifest.json`), `kind` (phase | dagger), `passed`, `segment` (full | interventions). A recovery item's
  scene is the evaluation scene it came from. `interventions` keeps the expert's span of a dagger episode: `recovery.segment` of the
  manifest entry, else the `expert` entries of `segments` in `<result stem>.interventions.json` next to the episode's result file (bags are never cut).
A new source (another run format) = one file in `plugins/sources/` or a module on `$RECIPE_PLUGIN_PATH`.

## Output (`plugins/outputs/`)
`selection`: `selection.json` (recipe settings, asked vs got per row and split, refused and left-out counts, the items per split with
absolute paths, segment and frames). A LeRobot export belongs to the dataset component and would be another output.
```
{recipe: {hash, seed, unit, fps, splits, on_short}, created,
 counts: [{source, kind, phase|null, asked: n|"all", got, available, splits: {name: got}}],
 refused: {source: n}, skipped: {source: {reason: n}},
 splits: {name: [{kind, source, phase, scene, config_seed, episode_dir, result, bag, segment: [t0, t1]|null, passed, task, robot,
                  duration_s, labels, frames}]}}
```

## Tests
```
./revolve test recipe        # unit per module + system tests on the tiny fake runs in tests/data
```
