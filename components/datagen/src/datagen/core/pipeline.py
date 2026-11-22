"""sample -> generate (validate each config first) -> record -> convert -> check, over injected plugins.

Every step reads and writes only the manifest, so each is idempotent and a killed run resumes: sampling is kept,
the walk skips decided seeds, record redoes episodes without a good recording, convert reruns when its episodes
change. Layout under `out`: manifest.json, report.md, <split>/scene_<seed>/scene.yaml (the scene instance, written by
`sample` when a randomiser varies the scene), <split>/episodes/seed_<n>/, dataset/. A recorder's Result.paths["episode"]
is the dir convert reads (default: <split>/episodes/seed_<n>).

`spec.run.workers` configs are generated (and episodes recorded) at a time; every decision is still taken in the
sequential order (pool.walk), so the manifest does not depend on it. A GUI run (not headless) has one window: 1 worker."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import manifest as mf
from . import pool, separation
from .errors import DatagenError
from .interrupt import Interrupted
from .pool import PoolExhausted
from .interfaces import (Check, Config, Converter, Generator, Instancer, Randomiser, Recorder, Result, Sampler,
                         Validator)
from .spec import DatagenSpec, SplitSpec, check_limits

STEPS = ("sample", "generate", "record", "convert", "check")


class PipelineError(DatagenError):
    pass


@dataclass
class PluginSet:
    samplers: Mapping[str, Sampler] = field(default_factory=dict)         # by strategy
    randomisers: Mapping[str, Randomiser] = field(default_factory=dict)   # by name
    instancer: Instancer | None = None            # writes the scene instance of every sampled config
    generator: Generator | None = None
    recorder: Recorder | None = None
    converter: Converter | None = None
    validators: Sequence[Validator] = ()
    checks: Sequence[Check] = ()


class Pipeline:
    def __init__(self, spec: DatagenSpec, plugins: PluginSet, out: Path, dry_run: bool = False,
                 log: Callable[[str], None] = print, abort: Callable[[], None] | None = None):
        """abort: stops what the running plugin calls started (their docker jobs); called when the run is interrupted."""
        check_limits(spec, {k: d for r in plugins.randomisers.values() for k, d in r.limits(spec).items()})
        self.spec, self.plugins, self.out, self.dry_run, self.log, self.abort = spec, plugins, out, dry_run, log, abort
        self.manifest = mf.Manifest.open(out / "manifest.json", spec, persist=not dry_run)
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
        if not self.dry_run:
            self.manifest.save()
            (self.out / "report.md").write_text(mf.report(self.spec, self.manifest.data))

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

    def episode_dir(self, config: Config) -> Path:
        return self.out / config.split / "episodes" / f"seed_{config.seed}"

    def scene_dir(self, config: Config) -> Path:
        """Where the config's scene instance lives (`<out>/<split>/scene_<seed>/scene.yaml`)."""
        return self.out / config.split / f"scene_{config.seed}"

    def _configs(self, split: SplitSpec) -> list[Config]:
        return [Config.from_dict(d) for d in self.manifest.split(split.name)["sampled"]]

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
            rec = self.manifest.split(s.name)
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
        self.manifest.save()

    def _write_instances(self, split: SplitSpec):
        """The scene instance of every sampled config that has none yet (a rejected one stays, to see why)."""
        if self.plugins.instancer is None or self.dry_run:
            return
        for c in self._configs(split):
            if not (self.scene_dir(c) / "scene.yaml").exists():
                self.plugins.instancer.write(c, self.scene_dir(c))

    def _generate(self, splits):
        gen = self._need("generator", self.plugins.generator)
        self._sample(splits)
        for s in splits:
            rec = self.manifest.split(s.name)

            def attempt(config):                       # on a worker thread: touches no manifest
                for v in self.plugins.validators:
                    res = self._call(f"validate {s.name} seed {config.seed}", v.validate, config)
                    if not res.ok:
                        return "validate", res
                return "generate", self._call(f"generate {s.name} seed {config.seed}", gen.generate, config,
                                              self.episode_dir(config))

            def accept(config, step, res):             # in walk order: reads the configs accepted before it
                if step == "generate" and res.ok and not res.realised and self.plugins.instancer:
                    res = replace(res, realised=self.plugins.instancer.realised(config))     # the sim starts at the instance
                if step == "generate" and (why := separation.too_close(s, self.spec.dims(), res, self.manifest.data)):
                    return "realised", Result.fail(why)       # the pool walk replaces it in its cell
                return step, res
            pool.walk(self._configs(s), rec, s.target, attempt, self.manifest.save, self.workers, accept, self.abort,
                      lambda left: [shutil.rmtree(self.episode_dir(c), ignore_errors=True) for c in left])
            self._said(f"generate {s.name}: {len(rec['used'])}/{s.target} accepted, {len(rec['rejected'])} rejected")

    def _record(self, splits):
        if self.plugins.recorder is None:
            return self.log("record: no recorder in the spec, skipped")
        for s in (s for s in splits if s.record):
            def call(e):                                # on a worker thread
                config = Config.from_dict(e["config"])
                return self._call(f"record {s.name} seed {config.seed}", self.plugins.recorder.record, config,
                                  Result.from_dict(e["generate"]), self.episode_dir(config))

            def commit(e, res):
                e["record"] = res.to_dict()
                self.manifest.save()
            todo = [e for e in self.manifest.split(s.name)["used"] if not e.get("record", {}).get("ok")]
            pool.ordered(todo, call, commit, self.workers, self.abort)
            recorded = sum(e.get("record", {}).get("ok", False) for e in self.manifest.split(s.name)["used"])
            self._said(f"record {s.name}: {recorded} recorded")

    def _convert(self, splits):
        if self.plugins.converter is None:
            return self.log("convert: no converter in the spec, skipped")
        episodes = [e["record"]["paths"].get("episode") or str(self.episode_dir(Config.from_dict(e["config"])))
                    for s in splits if s.record for e in self.manifest.split(s.name)["used"] if e.get("record", {}).get("ok")]
        done = self.manifest.data.get("convert", {})
        if done.get("episodes") == episodes and done["result"]["ok"]:
            return self.log("convert: up to date")
        res = self._call(f"convert {len(episodes)} episodes", self.plugins.converter.convert,
                         [Path(p) for p in episodes], self.out / "dataset")
        self.manifest.data["convert"] = {"episodes": episodes, "result": res.to_dict()}
        self.manifest.save()
        self._said(f"convert: {'ok' if res.ok else 'FAILED ' + res.reason}")
        return res.ok

    def _check(self, splits):
        if self.dry_run:                    # nothing ran, so there is nothing to judge
            return self.log(f"[dry-run] check {', '.join(c.name for c in self.plugins.checks)}")
        results = [(c.name, c.run(self.manifest.data)) for c in self.plugins.checks]
        self.manifest.data["checks"] = [{"name": n, **r.to_dict()} for n, r in results]
        self.manifest.save()
        for n, r in results:
            self.log(f"check {n}: {'PASS' if r.ok else 'FAIL ' + r.reason}")
        return all(r.ok for _, r in results)

    @staticmethod
    def _need(kind, plugin):
        if plugin is None:
            raise PipelineError(f"this step needs a {kind} plugin")
        return plugin
