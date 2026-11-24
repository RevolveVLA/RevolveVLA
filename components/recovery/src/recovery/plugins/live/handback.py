"""The hand-back half of live dagger (control going back to the policy): its spec, when the expert is done, and the measure of the
jump at the switch. Pure, no ROS.

Spec (`live: {handback: ..., max_interventions: n}`): `none` (default) = one-way, the expert finishes the task; or
`{when: phase_done | after_s, after_s: 3.0, settle_s: 0.3}`:
  phase_done  the evaluator's phase has advanced past the phase where the takeover happened,
  after_s     the expert has driven `after_s` sim seconds,
  settle_s    no new takeover for this long after the hand-back (the policy gets time to act; the supervisor's debounce restarts)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from datagen.core.spec import SpecError

WHENS = ("phase_done", "after_s")
KEYS = ("when", "after_s", "settle_s")
JUMP_HZ, JUMP_WINDOW_S = 30.0, 1.0          # the jump is measured on the arm's 30 Hz steps of the first second after a switch


@dataclass(frozen=True)
class Handback:
    when: str = "phase_done"
    after_s: float = 3.0
    settle_s: float = 0.3

    def reason(self, phases: Sequence[str], taken: str, now_phase: str | None, driven_s: float) -> str | None:
        """Why the expert (which took over in phase `taken`, `driven_s` sim seconds ago) is done, else None."""
        if self.when == "after_s":
            return f"after_s: {driven_s:.1f} s" if driven_s >= self.after_s else None
        if now_phase in phases and taken in phases and phases.index(now_phase) > phases.index(taken):
            return f"phase_done: {taken} -> {now_phase}"
        return None


def parse(v: Any, at: str = "live.handback") -> Handback | None:
    """`none` / absent -> None (one-way); else the mapping as a Handback."""
    if v is None or v == "none":
        return None
    if not isinstance(v, Mapping):
        raise SpecError(f"{at}: expected `none` or a mapping {{when: {' | '.join(WHENS)}, after_s, settle_s}}, got {v!r}")
    if unknown := sorted(set(v) - set(KEYS)):
        raise SpecError(f"{at}.{unknown[0]}: unknown key (allowed: {', '.join(KEYS)})")
    when = v.get("when", "phase_done")
    if when not in WHENS:
        raise SpecError(f"{at}.when: expected one of {WHENS}, got {when!r}")
    out = {}
    for key, low in (("after_s", 0.0), ("settle_s", -1.0)):
        if key in v:
            x = v[key]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or x <= low:
                raise SpecError(f"{at}.{key}: expected a number {'> 0' if low == 0 else '>= 0'}, got {x!r}")
            out[key] = float(x)
    if "after_s" in v and when != "after_s":
        raise SpecError(f"{at}.after_s: only for `when: after_s`")
    return Handback(when, **out)


class Jumps:
    """The largest step of the arm's joints (rad) between samples 1/JUMP_HZ of sim time apart, over the JUMP_WINDOW_S after
    `arm()`: what a switch of controller does to the arm (the policy's first actions after a hand-back). `peak` None until two samples."""

    def __init__(self, joints: Callable[[], Mapping[str, float] | None], names: Sequence[str], now: Callable[[], float]):
        self.joints, self.names, self.now = joints, tuple(names), now
        self.until, self.last, self.peak = -1.0, None, None

    def arm(self) -> None:
        self.until, self.last, self.peak = self.now() + JUMP_WINDOW_S, None, None

    @property
    def watching(self) -> bool:
        return self.until >= 0

    def sample(self) -> None:
        t = self.now()
        if t > self.until:
            self.until = -1.0
            return
        if self.last is not None and t - self.last[0] < 1 / JUMP_HZ - 1e-9:
            return
        q = self.joints()
        if not q:
            return
        q = [q[j] for j in self.names if j in q]
        if self.last is not None and q:
            step = max(abs(a - b) for a, b in zip(q, self.last[1]))
            self.peak = step if self.peak is None else max(self.peak, step)
        self.last = (t, q)
