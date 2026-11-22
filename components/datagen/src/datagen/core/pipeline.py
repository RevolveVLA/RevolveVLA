"""sample -> generate (validate each config, then run its scene instance on the sim) -> convert -> check, over injected plugins.

Every step reads and writes only the state, so each is idempotent and a killed run resumes: sampling is kept, the walk
skips decided seeds, a scene instance resumes from its own log (core/episodes.py). Layout under `out`: state.json (resumable
state), manifest.json (DATAGEN-v2.md 8.4, derived from the scene logs at the end of every run, also an interrupted one),
report.md, <split>/scene_<seed>/{scene.yaml, datagen.json, episode_<n>.json, episode_<n>_bag/}, logs/ (sim + client logs).
A scene is one pool-walk pick: accepted when its sim episodes gave `spec.episodes.per_scene` evaluator-accepted ones, rejected
(with the reason) otherwise; a split with `record: false` is only sampled and validated. `convert` is a seam: no Rosetta
contract for the new bags yet, so it is skipped.

`spec.run.workers` scenes run at a time (each its own sim + generator pair); every decision is still taken in the
sequential order (pool.walk), so the state does not depend on it. A GUI run (not headless) has one window: 1 worker."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import episodes, manifest as mf, pool, separation, stats
from . import state as st
from .episodes import SceneError
from .errors import DatagenError
from .interrupt import Interrupted
from .pool import PoolExhausted
from .interfaces import Check, Config, Converter, Generator, Instancer, Randomiser, Result, Runner, Sampler, Validator
from .spec import DatagenSpec, SplitSpec, check_limits

STEPS = ("sample", "generate", "convert", "check")
NO_CONVERT = "convert: skipped, no Rosetta contract for the new bags yet (DATAGEN-v2.md 7)"


class PipelineError(DatagenError):
    pass


@dataclass
class PluginSet:
    samplers: Mapping[str, Sampler] = field(default_factory=dict)         # by strategy
    randomisers: Mapping[str, Randomiser] = field(default_factory=dict)   # by name
    instancer: Instancer | None = None            # writes the scene instance of every sampled config
    generator: Generator | None = None
    runner: Runner | None = None
    converter: Converter | None = None
    validators: Sequence[Validator] = ()
    checks: Sequence[Check] = ()


class Pipeline:
    def __init__(self, spec: DatagenSpec, plugins: PluginSet, out: Path, dry_run: bool = False,
                 log: Callable[[str], None] = print, abort: Callable[[], None] | None = None,
                 scenes: Mapping[int, Path] | None = None):
        """abort: stops what the running plugin calls started (their sims); called when the run is interrupted.
        scenes: {seed: scene.yaml} of instances made elsewhere (a DATAGEN-v2.md 8.2 job): split `job`, no sampling."""
        check_limits(spec, {k: d for r in plugins.randomisers.values() for k, d in r.limits(spec).items()})
        self.spec, self.plugins, self.out, self.dry_run, self.log, self.abort = spec, plugins, out, dry_run, log, abort
        self.state = st.State.open(out / "state.json", spec, persist=not dry_run)
        self.scenes = dict(scenes or {})
        for s in spec.splits if self.scenes else ():
            rec = self.state.split(s.name)
            rec["sampled"] = rec["sampled"] or [Config(s.name, seed).to_dict() for seed in self.scenes]
        self.workers = spec.run.workers if spec.run.headless else 1
        self._sampled = set()
        if self.workers != spec.run.workers:
            log(f"run.headless false: one GUI window, so workers {spec.run.workers} -> 1")

    def run(self, steps: Sequence[str] = STEPS, splits: Sequence[str] | None = None) -> bool:
        """Run `steps` on `splits` (default all); False when a check failed."""
        known = [s.name for s in self.spec.splits]
        if unknown := [n for n in splits or () if n not in known]:
            raise PipelineError(f"unknown split {unknown} (splits: {known})")
        chosen = [s for s in self.spec.splits if splits is None or s.name in splits]
        ok = True
        try:
            for step in steps:
                ok &= getattr(self, f"_{step}")(chosen) is not False
        except (PoolExhausted, Interrupted):
            self._report()                  # the exhausted cells / what was decided are in the report; the error ends the run
            raise
        self._report()
        return ok

    def _report(self):
        """state.json, manifest.json (merged from the scene logs, in split + seed order) and report.md."""
        if self.dry_run:
            return
        self.state.save()
        dirs = [self.scene_dir(Config(s.name, seed)) for s in self.spec.splits if s.record
                for seed in sorted(stats.seeds(stats.rec(self.state.data, s.name)))]
        path = self.out / "manifest.json"
        data = mf.build(self.out, self.spec.task, self.spec.robot, dirs, mf.created(path))
        mf.write(path, data)
        faults = sum(len(episodes.read(d)["faults"]) for d in dirs)
        counts = data["counts"]
        (self.out / "report.md").write_text(st.report(self.spec, self.state.data) + (
            f"\n{counts['passed']}/{counts['attempted']} episodes accepted in {counts['scenes']} scenes, "
            f"{faults} infrastructure relaunches\n" if counts["attempted"] else ""))

    def _call(self, what: str, fn, *args):
        """Plugin call; a dry run only says what it would do."""
        if self.dry_run:
            self.log(f"[dry-run] {what}")
            return Result()
        return fn(*args)

    def _said(self, line: str):
        """A step's result line: a dry run did nothing, so it has none."""
        if not self.dry_run:
            self.log(line)

    def scene_dir(self, config: Config) -> Path:
        """Where the config's scene instance lives (`<out>/<split>/scene_<seed>/`; a job's instance: its own folder) and
        where its sim writes: the instance's `scene.yaml`, `datagen.json`, the episodes."""
        return self.scenes[config.seed].parent if config.seed in self.scenes else self.out / config.split / f"scene_{config.seed}"

    def _configs(self, split: SplitSpec) -> list[Config]:
        return [Config.from_dict(d) for d in self.state.split(split.name)["sampled"]]

    def _avoid(self, split: SplitSpec) -> list[Config]:
        """The configs another split's min_distance keeps this one away from (sampled first if need be)."""
        md = split.min_distance
        if md is None or md.source == split.name:
            return []
        source = self.spec.split(md.source)
        self._sample([source])
        return self._configs(source)

    def _sample(self, splits):
        for s in splits:
            rec = self.state.split(s.name)
            if not rec["sampled"]:
                sampler = self.plugins.samplers[s.sampler.strategy]
                configs = sampler.sample(s, self.spec.dims(), s.pool.seeds(), self.spec.seed, avoid=self._avoid(s))
                rec["sampled"] = [c.to_dict() for c in configs]
                if sampler.redrawn:
                    rec["redrawn"] = list(sampler.redrawn)
            self._write_instances(s)
            if s.name not in self._sampled:
                self._sampled.add(s.name)
                self.log(f"sample {s.name}: {len(rec['sampled'])} configs")
        self.state.save()

    def _write_instances(self, split: SplitSpec):
        """The scene instance of every sampled config that has none yet (a rejected one stays, to see why)."""
        if self.plugins.instancer is None or self.dry_run:
            return
        for c in self._configs(split):
            if not (self.scene_dir(c) / "scene.yaml").exists():
                self.plugins.instancer.write(c, self.scene_dir(c))

    def _scene(self, config: Config, generator: Generator) -> Result:
        """Run the config's scene instance on the sim -> accepted (enough evaluator-accepted episodes) or why not."""
        folder = self.scene_dir(config)
        try:
            self._need("runner", self.plugins.runner).run(config.seed, folder / "scene.yaml", generator)
        except SceneError as e:
            return Result.fail(str(e))
        return episodes.verdict(episodes.read(folder), self.spec.episodes)

    def _paths(self, config: Config) -> dict:
        return {"scene": os.path.relpath(self.scene_dir(config) / "scene.yaml", self.out)}

    def _generate(self, splits):
        gen = self._need("generator", self.plugins.generator)
        self._sample(splits)
        for s in splits:
            rec = self.state.split(s.name)

            def attempt(config):                       # on a worker thread: touches no state
                for v in self.plugins.validators:
                    res = self._call(f"validate {s.name} seed {config.seed}", v.validate, config)
                    if not res.ok:
                        return "validate", res
                if not s.record:
                    return "validate", Result(paths=self._paths(config))
                res = self._call(f"generate {s.name} seed {config.seed}", self._scene, config, gen)
                return "generate", replace(res, paths=self._paths(config)) if res.ok else res

            def accept(config, step, res):             # in walk order: reads the configs accepted before it
                if res.ok and not res.realised and self.plugins.instancer:
                    res = replace(res, realised=self.plugins.instancer.realised(config))     # the sim starts at the instance
                if res.ok and (why := separation.too_close(s, self.spec.dims(), res, self.state.data)):
                    return "realised", Result.fail(why)       # the pool walk replaces it in its cell
                return step, res
            pool.walk(self._configs(s), rec, s.target, attempt, self.state.save, self.workers, accept, self.abort)
            self._said(f"generate {s.name}: {len(rec['used'])}/{s.target} accepted, {len(rec['rejected'])} rejected")

    def _convert(self, splits):
        self.log(NO_CONVERT if self.plugins.converter else "convert: no converter in the spec, skipped")

    def _check(self, splits):
        if self.dry_run:                    # nothing ran, so there is nothing to judge
            return self.log(f"[dry-run] check {', '.join(c.name for c in self.plugins.checks)}")
        results = [(c.name, c.run(self.state.data)) for c in self.plugins.checks]
        self.state.data["checks"] = [{"name": n, **r.to_dict()} for n, r in results]
        self.state.save()
        for n, r in results:
            self.log(f"check {n}: {'PASS' if r.ok else 'FAIL ' + r.reason}")
        return all(r.ok for _, r in results)

    @staticmethod
    def _need(kind, plugin):
        if plugin is None:
            raise PipelineError(f"this step needs a {kind} plugin")
        return plugin
