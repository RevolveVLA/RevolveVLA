"""live dagger: the policy drives, a supervisor over privileged state (the `triggers` of `revolve_common.supervision`, fed by
/evaluator/state) watches it and, at the first firing, an expert takes over and finishes the task (one-way: control never goes
back) or, with `handback`, drives until its hand-back condition and gives control back to the policy, up to
`max_interventions` times per episode (state machine below). The check runs before each action of the policy, so the takeover tick is the first expert action and the rest of the
policy's chunk is dropped; the policy stops publishing and the expert starts from the measured joints. The bag stays whole; the
flags go to the episode's sidecar (core/flags.py).

Params (yaml `live: {plugin: dagger, ...}`): `triggers` (required: supervision trigger specs, e.g. phase_time / stall / condition),
`expert` (default `{plugin: moveit_takeover}`: registry kind `expert`), `record` (full | interventions: with `interventions` the
manifest keeps only the expert's span of the episode, the sidecar's `segment`; default full), `quota` {phase: takeovers wanted}
(a phase at its quota, or not listed once a quota is given, gets no more takeovers: the evaluator's verdict stands; default none),
`handback` (default `none`; see handback.py: `{when: phase_done | after_s, after_s, settle_s}`), `max_interventions` (takeovers per
episode; default 1 one-way, 3 with a handback; after the last one the policy runs on to the verdict).

State machine of an episode (one-way = the first expert state never ends):
  POLICY --trigger fires--> EXPERT --hand-back condition--> POLICY (supervisor re-armed, quiet for settle_s) --trigger--> EXPERT ...
  POLICY, after `max_interventions` takeovers: no more supervision; EXPERT ends by itself when the result is out (a verdict, or the
  task's last phase done), by a failure of the expert (no hand-back), or by the hand-back condition via the expert's stop hook.
The policy's drive is called again after each hand-back: it must start from the current observation (a fresh chunk, no stale queue)."""
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
from . import handback as hb
from .ros_state import RosState

PARAMS = ("triggers", "expert", "record", "quota", "handback", "max_interventions")
RECORDS = ("full", "interventions")


@dataclass
class Takeover:
    """One intervention: the firing (`t` supervisor time = s since its first state, `sim_t` the sim clock at the state that fired)
    and, once the expert handed back, `end_*` (None while it drives or when it never hands back)."""
    trigger: str
    reason: str
    phase: str
    t: float
    sim_t: float
    start_now: float = 0.0              # sim clock when the expert started
    end_t: float | None = None
    end_sim_t: float | None = None
    end_reason: str | None = None
    delta_in: float | None = None       # max_delta_action at the takeover (the expert's first second)
    delta_out: float | None = None      # ... and at the hand-back (the policy's first second)
    error: str | None = None


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
        self.handback = hb.parse(params.get("handback"))
        n = params.get("max_interventions")
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
            raise SpecError(f"live.max_interventions: expected an int >= 1, got {n!r}")
        if n is not None and n > 1 and self.handback is None:
            raise SpecError("live.max_interventions: more than 1 needs a `handback` (without one the expert never gives control back)")
        self.max_interventions = n or (3 if self.handback else 1)
        self.taken: Takeover | None = None          # the first intervention of the episode
        self.takes: list[Takeover] = []
        self.pending: Takeover | None = None        # the one whose expert drives (or is about to)
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
        self.taken = self.pending = self.error = None
        self.takes, self.quiet_until, self.last_t = [], 0.0, 0.0
        self.jumps = hb.Jumps(self.source.joints, arm_of(self.spec.robot, specs.REPO_SPECS).joints, ctx.client.now)
        self.expert.prepare(ctx)
        ctx.gate.check = self.check
        ctx.stop.check = self.expert_done

        def drive(n: int) -> Mapping[str, Any]:
            extra = dict(policy_drive(n))
            while self.pending is not None and not ctx.client.has_result(n):
                extra = {**extra, **self.take_over()}
                if not self.hand_back(n):
                    break
                again = policy_drive(n)
                extra = {**extra, **again, "steps": extra.get("steps", 0) + again.get("steps", 0)}
            return extra
        return drive

    def finish(self, episode: Mapping[str, Any]) -> Mapping[str, Any]:
        result = episode["result"]
        end = float(result["duration_s"])
        if not self.takes:
            return {"takeover_t": None, "takeover_sim_t": None, "trigger": None, "reason": None, "phase": None,
                    "max_delta_action": None, "expert": None, "expert_failure": False, "segments": [[0.0, end, "policy"]],
                    "segment": None}
        start = result.get("sim_time_start")
        at = lambda t, sim_t: round(min(max(sim_t - start, 0.0), end) if start is not None else min(t, end), 3)
        segments, spans, cursor = [], [], 0.0
        for tk in self.takes:
            a = at(tk.t, tk.sim_t)
            b = end if tk.end_sim_t is None else at(tk.end_t, tk.end_sim_t)
            if a > cursor:
                segments.append([cursor, a, "policy"])
            segments.append([a, b, "expert"])
            spans.append([a, b])
            cursor = b
        if cursor < end:
            segments.append([cursor, end, "policy"])
        failed = [bool(tk.error) or (tk.end_sim_t is None and not result["passed"]) for tk in self.takes]
        first = self.takes[0]
        flags = {"takeover_t": spans[0][0], "takeover_sim_t": round(first.sim_t, 3), "trigger": first.trigger,
                 "reason": first.reason, "phase": first.phase, "max_delta_action": first.delta_in, "expert": self.notes.get("expert"),
                 "expert_failure": any(failed), "segments": segments,
                 "segment": spans[0] if self.record == "interventions" and len(spans) == 1 else None}
        if self.handback:
            flags["kept"] = spans if self.record == "interventions" else None
            flags["interventions"] = [
                {"trigger": tk.trigger, "reason": tk.reason, "phase": tk.phase, "takeover_t": a, "handback_t": None if tk.end_sim_t is None else b,
                 "handback_reason": tk.end_reason, "max_delta_action": {"takeover": tk.delta_in, "handback": tk.delta_out},
                 "expert_failure": bad} for tk, (a, b), bad in zip(self.takes, spans, failed)]
        return flags

    # ---- one episode
    def check(self) -> bool:
        """Polled before each action of the policy: True while an intervention is pending (latched until the hand-back, if any)."""
        if self.pending is not None:
            return True
        if self.jumps.watching:
            self.jumps.sample()
            self.watch_out()
        if len(self.takes) >= self.max_interventions:
            return False
        tick = self.source.tick()
        if tick is None or tick.phase in self.ctx.closed:
            return False
        self.last_t = tick.t
        if tick.t < self.quiet_until:
            return False
        firing = self.supervisor.check(tick)
        if firing is None:
            return False
        state = self.ctx.client.state()
        self.pending = Takeover(firing.trigger, firing.reason, firing.phase, firing.t, state["t"] if state else self.ctx.client.now())
        self.takes.append(self.pending)
        self.taken = self.taken or self.pending
        return True

    def watch_out(self) -> None:
        """The jump of the policy's first second after the last hand-back, as it is measured."""
        if self.takes and self.takes[-1].end_sim_t is not None:
            self.takes[-1].delta_out = self.jumps.peak

    def expert_done(self) -> bool:
        """The expert's stop hook (polled by it between and during its motion segments): the hand-back condition holds."""
        tk = self.pending
        if tk is None or self.handback is None:
            return False
        if tk.end_reason is None:
            state = self.ctx.client.state()
            tk.end_reason = self.handback.reason([p.name for p in load_task(self.spec).task.phases], tk.phase,
                                                 state and state.get("phase"), self.ctx.client.now() - tk.start_now)
        return tk.end_reason is not None

    def hand_back(self, n: int) -> bool:
        """After the expert returned: give control back when it stopped on the hand-back condition (and the episode runs on)."""
        tk = self.pending
        if tk.end_reason is None or self.ctx.client.has_result(n):
            tk.end_reason = None
            return False
        state, tick = self.ctx.client.state(), self.source.tick()
        tk.end_sim_t, tk.end_t = state["t"] if state else self.ctx.client.now(), tick.t if tick else self.last_t
        self.quiet_until = tk.end_t + self.handback.settle_s
        self.pending = None
        self.supervisor.start()
        self.jumps.arm()
        return True

    def take_over(self) -> Mapping[str, Any]:
        t = self.pending
        t.start_now = self.ctx.client.now()
        try:
            self.notes = dict(self.expert.takeover(self.ctx, f"{t.trigger}: {t.reason}"))
        except Exception as e:                      # a failed expert ends the try (the evaluator times it out), not the run
            traceback.print_exc(file=sys.stderr)
            self.error = t.error = f"{type(e).__name__}: {e}"
            self.notes = {}
            t.end_reason = None
        t.delta_in = self.notes.get("max_delta_action")
        return {"expert": self.notes.get("expert"), "notes": self.notes.get("notes", self.error)}
