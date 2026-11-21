"""Fake datagen plugins for the CPU tests (no docker, no GPU): tiny files, scripted failures, a call log.

Registered under the module's own name, so a spec says `plugin: fake_datagen` (put tests/ on $DATAGEN_PLUGIN_PATH).
Params: generator / recorder `fail_seeds`, validator `reject_seeds`, generator `realised` {seed: x} (the realised
`fake_datagen.x` of that seed; others report only `x`). `KILL_AFTER = n` makes the generator raise
KeyboardInterrupt on its (n+1)-th call (a killed run; a spec param would change the spec hash). A subprocess (the CLI
system tests) gets both through the environment: $FAKE_DATAGEN_KILL_AFTER, and its calls appended to $FAKE_DATAGEN_LOG."""
import json
import os
from pathlib import Path

from datagen.core.interfaces import Converter, Generator, Randomiser, Recorder, Result, Validator
from datagen.core.registry import register
from datagen.core.spec import Range

KILL_AFTER = int(os.environ["FAKE_DATAGEN_KILL_AFTER"]) if "FAKE_DATAGEN_KILL_AFTER" in os.environ else None
CALLS = []      # ("generate"|"record"|"convert"|"validate", seed) in call order


def called(kind, n):
    CALLS.append((kind, n))
    if log := os.environ.get("FAKE_DATAGEN_LOG"):
        with open(log, "a") as f:
            f.write(f"{kind} {n}\n")


@register("randomiser", "fake_datagen")
class Knob(Randomiser):
    def limits(self, spec):
        return {"fake_datagen.x": Range(-1.0, 1.0), "fake_datagen.y": Range(0.0, 10.0)}

    def realise(self, values, context):
        return {"offset": values["x"]}


@register("generator", "fake_datagen")
class FakeGenerator(Generator):
    def __init__(self, spec, params):
        self.fail, self.n, self.realised = set(params.get("fail_seeds", [])), 0, params.get("realised", {})

    def generate(self, config, out):
        called("generate", config.seed)
        self.n += 1
        if KILL_AFTER is not None and self.n > KILL_AFTER:
            raise KeyboardInterrupt("simulated kill")
        if config.seed in self.fail:
            return Result.fail("planner found no path")
        out.mkdir(parents=True, exist_ok=True)
        (out / "episode.json").write_text(json.dumps(dict(config.values)))
        realised = {"x": config.values.get("fake_datagen.x")}
        if config.seed in self.realised:
            realised["fake_datagen.x"] = self.realised[config.seed]
        return Result(realised=realised, paths={"episode": str(out / "episode.json")})


@register("recorder", "fake_datagen")
class FakeRecorder(Recorder):
    def __init__(self, spec, params):
        self.fail = set(params.get("fail_seeds", []))

    def record(self, config, generated, out):
        called("record", config.seed)
        if config.seed in self.fail:
            return Result.fail("replay diverged")
        (out / "bag.txt").write_text("bag")
        return Result(paths={"bag": str(out / "bag.txt")})


@register("converter", "fake_datagen")
class FakeConverter(Converter):
    def __init__(self, spec, params):
        pass

    def convert(self, episodes, out):
        called("convert", len(episodes))
        out.mkdir(parents=True, exist_ok=True)
        (out / "dataset.json").write_text(json.dumps([Path(e).name for e in episodes]))
        return Result(paths={"dataset": str(out / "dataset.json")})


@register("validator", "fake_datagen")
class FakeValidator(Validator):
    def __init__(self, spec, params):
        self.reject = set(params.get("reject_seeds", []))

    def validate(self, config):
        called("validate", config.seed)
        return Result.fail("too close to the wall") if config.seed in self.reject else Result()
