"""Fake evaluation plugins for the CPU tests (no docker, no GPU, no checkpoint): tiny files and scripted verdicts.

Registered under the module's own name, so a spec says `source: fake_eval` / `plugin: fake_eval` (put tests/ on
$EVAL_PLUGIN_PATH). Set: one config per seed of the pool, start dirs under `<out>/<set>/starts/`; its check fails when
$FAKE_EVAL_FAIL_CHECK is set. Rollout: record `{passed: seed even, dir, ckpt}`, its calls appended to $FAKE_EVAL_LOG.
Judge: passes the record's `passed`, except for the seeds in its `fail_seeds` param."""
import os
from pathlib import Path

from datagen.core.interfaces import Config, Result

from evaluation.core.interfaces import Built, CkptSource, EvalItem, EvalSet, Judge, Policy, ResolvedCkpt, Rollout, TrainingRun, TrainingSource, Verdict
from evaluation.core.registry import REGISTRY


def logged(line):
    if log := os.environ.get("FAKE_EVAL_LOG"):
        with open(log, "a") as f:
            f.write(line + "\n")


@REGISTRY.register("ckpt", "fake_eval")
class FakeCkpt(CkptSource):
    def __init__(self, spec, params):
        self.params = params

    def resolve(self, out):
        return ResolvedCkpt(str(out / "ckpt"), f"fake {self.params.get('tag', '')}".strip())


@REGISTRY.register("training", "fake_eval")
class FakeTraining(TrainingSource):
    def __init__(self, spec, params):
        pass

    def load(self, root):
        return TrainingRun(root / "fake_training", None)


@REGISTRY.register("set", "fake_eval")
class FakeSet(EvalSet):
    def __init__(self, spec):
        pass

    def build(self, spec, out, workers, abort=None):
        seeds = (spec.pool.seeds() if spec.pool else range(spec.count))[:spec.count]
        items = []
        for s in seeds:
            d = out / spec.name / "starts" / f"seed_{s}"
            d.mkdir(parents=True, exist_ok=True)
            items.append(EvalItem(Config(spec.name, s), d))
        fail = "FAKE_EVAL_FAIL_CHECK" in os.environ
        return Built(items, {"fake": Result.fail("scripted") if fail else Result()})


@REGISTRY.register("policy", "fake_eval")
class FakePolicy(Policy):
    name = "fake_eval"

    def __init__(self, spec, params):
        self.needs_ckpt = params.get("needs_ckpt", False)


@REGISTRY.register("rollout", "fake_eval")
class FakeRollout(Rollout):
    def __init__(self, spec, params):
        pass

    def run(self, item, out, *, seed, runner, policy, ckpt, gui=False):
        out.mkdir(parents=True, exist_ok=True)
        logged(f"rollout {item.config.split} {seed} {out.name}")
        return {"passed": seed % 2 == 0, "dir": str(out), "ckpt": ckpt.path if ckpt else None}


@REGISTRY.register("judge", "fake_eval")
class FakeJudge(Judge):
    def __init__(self, spec, params):
        self.fail = set(params.get("fail_seeds", []))

    def score(self, record, rollout_dir, task):
        return Verdict(record["passed"] and int(Path(record["dir"]).name.split("_")[0]) not in self.fail)
