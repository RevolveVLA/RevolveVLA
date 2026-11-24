"""The live recorder inside a rollout's client container: `python3 -m evaluation.rollout_client ... --driver
recovery.live_client:make --driver-args <JSON>` (the evaluation's policy client with its policy wrapped). `make` builds the
`Live` the client wants (evaluation.rollout_client.Live) from the recovery spec's `live:` plugin; `finish` writes the episode's
sidecar (core/flags.py). driver-args: {spec: the recovery spec as JSON, closed: [phases without takeover], perturb: how a test
policy alters its actions (plugins/policies) | null}."""
from __future__ import annotations

import importlib
from pathlib import Path

from .core import flags
from .core.interfaces import Gate, LiveContext
from .core.registry import REGISTRY
from .core.spec import parse


class RecorderLive:
    """The client's view of a live recorder: the recorder's driver / flags and the policy's alteration, as the client's hooks."""

    def __init__(self, recorder, ctx: LiveContext, perturb=None):
        self.recorder, self.ctx, self.perturb = recorder, ctx, perturb

    def halt(self) -> bool:
        return self.ctx.gate()

    def series(self, series):
        return self.perturb(series) if self.perturb else series

    def driver(self, drive):
        return self.recorder.driver(drive, self.ctx)

    def finish(self, episode) -> None:
        flags.write(self.ctx.out, episode["episode"], self.recorder.finish(episode))


def perturbation(p):
    """`{name, ...params}` -> the function of that test policy (plugins/policies/<name>.py `perturb(series, params)`)."""
    if not p:
        return None
    module = importlib.import_module(f"recovery.plugins.policies.{p['name']}")
    return lambda series: module.perturb(series, p)


def make(client, args, instance, rollout) -> RecorderLive:
    """The `Live` of one episode (called by the policy client once the sim is ready)."""
    spec = parse(args["spec"], "<spec>")
    recorder = REGISTRY.create("live", spec.live.plugin, spec, spec.live.params)
    ctx = LiveContext(client, spec.task, spec.robot, Path(instance), Path(client.out), Gate(), frozenset(args.get("closed", ())))
    return RecorderLive(recorder, ctx, perturbation(args.get("perturb")))
