"""The Rosetta recorder: the camera replay + bag of the generator's accepted attempt (generate.record_attempt), so an
episode is recorded only after its camera-off check passed and its replay reproduces the check's state log.

Needs a generator that ends at the accepted check in the same dir (automoma: `generate(record=False)`): Result.realised
carries its `attempt_id`. Result.paths: `episode` (the episode dir the dataset component reads: bag/, episode.json with the
`datagen` block, plan.json, state_log.npz) and `bag`. Resume: an accepted recording in the dir is kept, not replayed again."""
from __future__ import annotations

from pathlib import Path

from datagen.core.interfaces import Config, Recorder, Result
from datagen.core.registry import register

from .. import generate_module


@register("recorder", "rosetta")
class RosettaRecorder(Recorder):
    def __init__(self, spec, params, jobs=None):
        """jobs: generate.Jobs (default: the docker jobs), injected by tests."""
        if params:
            raise ValueError(f"recorder rosetta: unknown params {sorted(params)}")
        self.spec, self.jobs = spec, jobs

    def record(self, config: Config, generated: Result, out: Path) -> Result:
        if "attempt_id" not in generated.realised:
            return Result.fail("the generated result has no attempt_id to record")
        G = generate_module.load(self.spec.robot, self.spec.task)
        try:
            e = G.record_attempt(self.spec.task, config.seed, generated.realised["attempt_id"], out,
                                 self.jobs or G.DOCKER, **generate_module.gui_kw(self.spec))
        except Exception as err:         # an infrastructure error fails this episode, not the run
            return Result.fail(repr(err))
        if not e["accepted"]:
            return Result.fail("; ".join(e["reasons"]))
        episode = Path(out) / e["dir"]
        return Result(True, "", {}, {"episode": str(episode), "bag": str(episode / "bag")})
