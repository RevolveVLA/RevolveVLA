"""Held-out set: `count` new scene instances drawn inside the training ranges (or the set's narrower ones), away from the
training configs. No generator is involved: an instance (`<out>/<set>/scene_<seed>/scene.yaml`, written by datagen's
instancer from the training randomisers) is the start a rollout begins from.

Drawing: datagen's sampler over the pool in the narrow dims, redrawing a config closer than `min_distance` to a training
config measured in the TRAINING dims (an instance's values are the realised start, so `min_distance.realised` is an alias).
Walking: datagen's pool walk takes the first `count` picks; one the training spec's validators reject (collision_free) is
replaced from its cell. The state is `<out>/<set>/state.json` (datagen's state shape), so a killed run resumes."""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from datagen.core import pool, registry
from datagen.core.interfaces import Config, Result
from datagen.core.samplers import distance
from datagen.core.state import State

from ...core import checks, sets
from ...core.interfaces import Built, EvalItem, EvalSet
from ...core.registry import REGISTRY


def instancer_for(spec):
    """datagen's: the scene instancer when a randomiser of the spec varies the scene, else None (no file is written)."""
    if any(registry.create("randomiser", n).scene_field for n in spec.randomisers):
        return registry.create("instancer", "scene_instance", spec, {})


@REGISTRY.register("set", "heldout")
class HeldoutSet(EvalSet):
    def __init__(self, spec):
        self.spec = spec

    def build(self, set_spec, out, workers, abort=None):
        spec, name, md = self.spec, set_spec.name, set_spec.min_distance
        training = sets.Training.load(spec, sets.repo_root(spec, out))
        derived = sets.derive(spec, set_spec)
        state = State.open(out / name / "state.json", derived)
        rec, base = state.split(name), out / name
        trained = {"splits": {md.source: training.rec(md.source)} if md else {}}      # the split it keeps away from
        if not rec["sampled"]:
            self._sample(set_spec, derived, training, rec)
            state.save()
        instancer = instancer_for(derived)
        validators = [registry.create("validator", v.plugin, derived, v.params) for v in spec.train.validators]

        def attempt(config):                                    # on a worker thread: touches no state
            for v in validators:
                if not (res := v.validate(config)).ok:
                    return "validate", res
            folder = base / f"scene_{config.seed}"
            res = instancer.write(config, folder) if instancer else Result()
            return "instance", dataclasses.replace(res, paths={"scene": os.path.relpath(folder / "scene.yaml", base)})

        pool.walk([Config.from_dict(c) for c in rec["sampled"]], rec, set_spec.count, attempt, state.save, workers,
                  abort=abort)
        view = {"splits": {**trained["splits"], name: rec}}
        results = checks.run_checks(checks.set_checks(spec, derived, name, training), view, training)
        return Built([self._item(e, base) for e in rec["used"]], results)

    @staticmethod
    def _item(entry, base: Path) -> EvalItem:
        return EvalItem(Config.from_dict(entry["config"]), base / entry["generate"]["paths"]["scene"])

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
