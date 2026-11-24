"""expert moveit_takeover: the datagen `moveit` expert taking the task over from the robot's current state (its `resume`: the entry
the state calls for, a smooth start from the measured joints), in the policy client's own process: its planner is started
before the episode (`prepare`), the takeover runs when the live recorder hands over. Params: the moveit generator's (`repo`)."""
from __future__ import annotations

import json
from typing import Any, Mapping

from datagen.core.errors import DatagenError
from datagen.core.spec import SpecError

from ...core.interfaces import LiveContext
from ...core.registry import REGISTRY


def max_delta_action(notes: Any) -> float | None:
    """The `max_delta_action` the expert logged for its first second (notes: the JSON list of its marks), else None."""
    try:
        return next(m["max_delta_action"] for m in json.loads(notes) if m.get("what") == "first_second")
    except (TypeError, ValueError, StopIteration, KeyError):
        return None


@REGISTRY.register("expert", "moveit_takeover")
class MoveItTakeover:
    def __init__(self, spec, params: Mapping[str, Any]):
        from datagen.plugins.moveit.generator import MoveItGenerator
        try:
            self.generator = MoveItGenerator(spec, {**params, "resume": True})
        except DatagenError as e:
            raise SpecError(f"live.expert moveit_takeover: {e}") from None

    def prepare(self, ctx: LiveContext) -> None:
        self.generator.prepare(ctx.client, ctx.instance, ctx.task, ctx.robot)

    def takeover(self, ctx: LiveContext, why: str) -> Mapping[str, Any]:
        done = self.generator.drive(ctx.client, ctx.instance, ctx.task, ctx.robot, stop=ctx.stop)
        return {**done, "max_delta_action": max_delta_action(done.get("notes"))}
