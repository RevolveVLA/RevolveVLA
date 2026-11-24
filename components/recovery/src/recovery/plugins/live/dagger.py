"""live dagger: the policy drives, a supervisor over privileged state (the `triggers` of `revolve_common.supervision`, fed by
/evaluator/state) watches it and, at the first firing, an expert takes over and finishes the task (one-way: control never goes
back). The check runs before each action of the policy, so the takeover tick is the first expert action and the rest of the
policy's chunk is dropped; the policy stops publishing and the expert starts from the measured joints. The bag stays whole; the
flags go to the episode's sidecar (core/flags.py).

Params (yaml `live: {plugin: dagger, ...}`): `triggers` (required: supervision trigger specs, e.g. phase_time / stall / condition),
`expert` (default `{plugin: moveit_takeover}`: registry kind `expert`), `record` (full | interventions: with `interventions` the
manifest keeps only the expert's span of the episode, the sidecar's `segment`; default full), `quota` {phase: takeovers wanted}
(a phase at its quota, or not listed once a quota is given, gets no more takeovers: the evaluator's verdict stands; default none)."""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Any, Mapping

from datagen.core.spec import SpecError
from revolve_common import specs
from revolve_common.supervision import LiveSource, build

from ...core.interfaces import Drive, LiveContext
from ...core.registry import REGISTRY
from ...core.robot import arm_of
from ...core.scan import load_task
from .ros_state import RosState

PARAMS = ("triggers", "expert", "record", "quota")
RECORDS = ("full", "interventions")


@dataclass(frozen=True)
class Takeover:
    trigger: str
    reason: str
    phase: str
    t: float            # supervisor time (s since its first state)
    sim_t: float        # the sim clock at the state that fired


@REGISTRY.register("live", "dagger")
class DAgger:
    def __init__(self, spec, params: Mapping[str, Any]):
        if unknown := sorted(set(params) - set(PARAMS)):
            raise SpecError(f"live.{unknown[0]}: unknown key (allowed: {', '.join(PARAMS)})")
        if not isinstance(params.get("triggers"), list) or not params["triggers"]:
            raise SpecError("live.triggers: expected a non-empty list of {plugin: ...} (what makes the expert take over)")
        self.spec, self.triggers = spec, params["triggers"]
        self.record = params.get("record", "full")
        if self.record not in RECORDS:
            raise SpecError(f"live.record: expected one of {RECORDS}, got {self.record!r}")
        quota = params.get("quota")
        if quota is not None and not (isinstance(quota, Mapping) and all(isinstance(n, int) and n >= 0 for n in quota.values())):
            raise SpecError(f"live.quota: expected {{phase: takeovers wanted (int >= 0)}}, got {quota!r}")
        self.quota = dict(quota) if quota is not None else None
        expert = params.get("expert", {"plugin": "moveit_takeover"})
        if not isinstance(expert, Mapping) or "plugin" not in expert:
            raise SpecError("live.expert: expected a mapping with a `plugin` key")
        self.expert_ref = (expert["plugin"], {k: v for k, v in expert.items() if k != "plugin"})
        self.taken: Takeover | None = None
        self.error: str | None = None
        self.notes: Mapping[str, Any] = {}

    # ---- seams: what the CPU tests replace
    def make_source(self, ctx: LiveContext) -> LiveSource:
        state = RosState(ctx.client.node, ctx.robot)
        return LiveSource(ctx.client, state.joints, state.base)

    def make_expert(self):
        return REGISTRY.create("expert", self.expert_ref[0], self.spec, self.expert_ref[1])

    def make_supervisor(self):
        try:
            return build(self.triggers, load_task(self.spec), arm_joints=arm_of(self.spec.robot, specs.REPO_SPECS).joints,
                         path="live.triggers")
        except specs.SpecError as e:
            raise SpecError(str(e)) from None

    # ---- LiveRecorder
    def validate(self, spec) -> None:
        task = load_task(spec)
        phases = [p.name for p in task.task.phases]
        if unknown := [p for p in self.quota or () if p not in phases]:
            raise SpecError(f"live.quota: unknown phase {unknown} (phases of {spec.task}: {', '.join(phases)})")
        self.spec = spec
        self.make_supervisor()
        self.make_expert()

    def closed(self, taken: Mapping[str, int]) -> frozenset[str]:
        if self.quota is None:
            return frozenset()
        phases = [p.name for p in load_task(self.spec).task.phases]
        return frozenset(p for p in phases if taken.get(p, 0) >= self.quota.get(p, 0))

    def driver(self, policy_drive: Drive, ctx: LiveContext) -> Drive:
        self.supervisor, self.source, self.expert, self.ctx = self.make_supervisor(), self.make_source(ctx), self.make_expert(), ctx
        self.taken = self.error = None
        self.expert.prepare(ctx)
        ctx.gate.check = self.check

        def drive(n: int) -> Mapping[str, Any]:
            extra = dict(policy_drive(n))
            if self.taken is None or ctx.client.has_result(n):
                return extra
            return {**extra, **self.take_over()}
        return drive

    def finish(self, episode: Mapping[str, Any]) -> Mapping[str, Any]:
        result = episode["result"]
        end = float(result["duration_s"])
        t = self.taken
        if t is None:
            return {"takeover_t": None, "takeover_sim_t": None, "trigger": None, "reason": None, "phase": None,
                    "max_delta_action": None, "expert": None, "expert_failure": False, "segments": [[0.0, end, "policy"]],
                    "segment": None}
        start = result.get("sim_time_start")
        at = min(max(t.sim_t - start, 0.0), end) if start is not None else min(t.t, end)
        return {"takeover_t": round(at, 3), "takeover_sim_t": round(t.sim_t, 3), "trigger": t.trigger, "reason": t.reason,
                "phase": t.phase, "max_delta_action": self.notes.get("max_delta_action"), "expert": self.notes.get("expert"),
                "expert_failure": bool(self.error) or not result["passed"],
                "segments": [*([[0.0, round(at, 3), "policy"]] if at > 0 else []), [round(at, 3), end, "expert"]],
                "segment": [round(at, 3), end] if self.record == "interventions" else None}

    # ---- one episode
    def check(self) -> bool:
        """Polled before each action of the policy: True once a trigger has fired (latched: the handover is one-way)."""
        if self.taken is not None:
            return True
        tick = self.source.tick()
        if tick is None or tick.phase in self.ctx.closed:
            return False
        firing = self.supervisor.check(tick)
        if firing is None:
            return False
        state = self.ctx.client.state()
        self.taken = Takeover(firing.trigger, firing.reason, firing.phase, firing.t, state["t"] if state else self.ctx.client.now())
        return True

    def take_over(self) -> Mapping[str, Any]:
        t = self.taken
        try:
            self.notes = dict(self.expert.takeover(self.ctx, f"{t.trigger}: {t.reason}"))
        except Exception as e:                      # a failed expert ends the try (the evaluator times it out), not the run
            traceback.print_exc(file=sys.stderr)
            self.error, self.notes = f"{type(e).__name__}: {e}", {}
        return {"expert": self.notes.get("expert"), "notes": self.notes.get("notes", self.error)}
