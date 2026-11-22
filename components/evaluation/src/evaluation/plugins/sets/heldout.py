"""Held-out set: `count` new configs drawn inside the training ranges (or the set's narrower ones), away from the
training configs, started by the training spec's own generator with `record: false` (AutoMoMa: the accepted check dir
is the start dir a rollout begins from).

Drawing: datagen's sampler over the pool in the narrow dims, redrawing a config closer than `min_distance` to a training
config measured in the TRAINING dims. Walking: datagen's pool walk generates the first `count` picks; a start whose
realised values are closer than that (`min_distance.realised`) is rejected and replaced from its cell. The state is
`<out>/<set>/manifest.json` (datagen's manifest shape), so a killed run resumes without generating a seed twice."""
from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

from datagen.core import pool, registry, separation
from datagen.core.interfaces import Config, Result
from datagen.core.state import State as Manifest        # datagen v2 renamed it (state.json); PRD 05 revisits
from datagen.core.samplers import distance

from ...core import checks, sets
from ...core.interfaces import Built, EvalItem, EvalSet
from ...core.registry import REGISTRY


@REGISTRY.register("set", "heldout")
class HeldoutSet(EvalSet):
    def __init__(self, spec):
        self.spec = spec

    def build(self, set_spec, out, workers, abort=None):
        spec, name, md = self.spec, set_spec.name, set_spec.min_distance
        training = sets.Training.load(spec, sets.repo_root(spec, out))
        if md and md.realised:
            training.need_manifest(f"sets.{name}.min_distance.realised")
        derived = sets.derive(spec, set_spec)
        manifest = Manifest.open(out / name / "manifest.json", derived)
        rec, base = manifest.split(name), out / name / "starts"
        trained = {"splits": {md.source: training.rec(md.source)} if md else {}}      # the split it keeps away from
        if not rec["sampled"]:
            self._sample(set_spec, derived, training, rec)
            manifest.save()
        generator = registry.create("generator", spec.train.generator.plugin, derived, spec.train.generator.params)
        split, dims = derived.split(name), spec.train.dims()

        def attempt(config):                                    # on a worker thread: touches no manifest
            return "generate", generator.generate(config, base / f"seed_{config.seed}")

        def accept(config, step, res):                          # in walk order
            if step == "generate" and (why := separation.too_close(split, dims, res, trained)):
                return "realised", Result.fail(why)
            return step, res
        def discard(left):                                      # an interrupted attempt's partial output
            for c in left:
                shutil.rmtree(base / f"seed_{c.seed}", ignore_errors=True)
        pool.walk([Config.from_dict(c) for c in rec["sampled"]], rec, set_spec.count, attempt, manifest.save, workers,
                  accept, abort, discard)
        view = {"splits": {**trained["splits"], name: rec}}
        results = checks.run_checks(checks.set_checks(spec, derived, name, training), view, training)
        return Built([self._item(e, base) for e in rec["used"]], results)

    @staticmethod
    def _item(entry, base: Path) -> EvalItem:
        """The start dir is the generator's accepted check dir (else the dir it generated into)."""
        config = Config.from_dict(entry["config"])
        return EvalItem(config, Path(entry["generate"].get("paths", {}).get("check") or base / f"seed_{config.seed}"))

    def _sample(self, set_spec, derived, training, rec):
        md = set_spec.min_distance
        avoid = training.configs(md.source) if md else []
        dims = self.spec.train.dims()

        def reject(config):                                     # the training dims, however narrow the drawing dims
            return "min_distance" if any(distance(config.values, t.values, dims) < md.value for t in avoid) else None
        split = dataclasses.replace(derived.split(set_spec.name), min_distance=None)    # `reject` measures instead
        sampler = registry.create("sampler", set_spec.sampler.strategy)
        configs = sampler.sample(split, sets.sample_dims(self.spec, set_spec), set_spec.pool.seeds(),
                                 self.spec.train.seed, reject=reject if md else None)
        rec["sampled"] = [c.to_dict() for c in configs]
        if sampler.redrawn:
            rec["redrawn"] = list(sampler.redrawn)
