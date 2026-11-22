"""Fake datagen plugins for the CPU tests (no docker, no GPU, no sim): scripted delays, a call log.

Registered under the module's own name, so a spec says `plugin: fake_datagen` (put tests/ on $DATAGEN_PLUGIN_PATH). The sim
is the `fake` runner (plugins/runners/fake.py: which scenes pass, which sims fault). Params: generator `delays` {seed:
seconds} (a slow expert; the scene dir exists while it sleeps), generator `delay` (seconds for every seed), validator
`reject_seeds`. `KILL_AFTER = n` makes the generator raise KeyboardInterrupt on its (n+1)-th drive (a killed run; a spec
param would change the spec hash). A subprocess (the CLI system tests) gets both through the environment:
$FAKE_DATAGEN_KILL_AFTER, and its calls appended to $FAKE_DATAGEN_LOG."""
import json
import os
import threading
import time
from pathlib import Path

from datagen.core.interfaces import Converter, Generator, Instancer, Randomiser, Result, Validator
from datagen.core.registry import register
from datagen.core.spec import Range

KILL_AFTER = int(os.environ["FAKE_DATAGEN_KILL_AFTER"]) if "FAKE_DATAGEN_KILL_AFTER" in os.environ else None
CALLS = []      # ("drive"|"convert"|"validate", seed) in call order
ACTIVE = {"now": 0, "peak": 0}      # generator drives running at once, and the most there were
_LOCK = threading.Lock()


def slow(kind, seed, seconds):
    """Sleep `seconds`, counting the call as running meanwhile."""
    with _LOCK:
        ACTIVE["now"] += 1
        ACTIVE["peak"] = max(ACTIVE["peak"], ACTIVE["now"])
    try:
        time.sleep(seconds)
    finally:
        with _LOCK:
            ACTIVE["now"] -= 1


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


def seed_of(instance: Path) -> int:
    return int(instance.parent.name.split("_")[1])


@register("generator", "fake_datagen")
class FakeGenerator(Generator):
    def __init__(self, spec, params):
        self.n = 0
        self.delay, self.delays = params.get("delay", 0), {int(k): v for k, v in params.get("delays", {}).items()}

    def drive(self, client, instance, task, robot):
        seed = seed_of(instance)
        called("drive", seed)
        self.n += 1
        if KILL_AFTER is not None and self.n > KILL_AFTER:
            raise KeyboardInterrupt("simulated kill")
        if seconds := self.delays.get(seed, self.delay):
            slow("drive", seed, seconds)
        return {"expert": "fake_v1", "notes": f"seed {seed}"}


class FakeInstancer(Instancer):
    """Scripted starts: `realised` {seed: {"fake_datagen.x": x}} (what the sim's start of that instance is)."""

    def __init__(self, realised):
        self.start = realised

    def write(self, config, out):
        return Result(realised=self.realised(config))

    def realised(self, config):
        return self.start.get(config.seed, {})


@register("converter", "fake_datagen")
class FakeConverter(Converter):
    def __init__(self, spec, params):
        pass

    def convert(self, episodes, out):
        called("convert", len(episodes))
        return Result()


@register("validator", "fake_datagen")
class FakeValidator(Validator):
    def __init__(self, spec, params):
        self.reject = set(params.get("reject_seeds", []))

    def validate(self, config):
        called("validate", config.seed)
        return Result.fail("too close to the wall") if config.seed in self.reject else Result()
