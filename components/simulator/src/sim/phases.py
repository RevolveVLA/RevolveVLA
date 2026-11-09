"""Live (online) phase validation of one fed action stream: the phase rules, the gates and the
report. Numpy only; phase_replay_system.py (in gz) feeds it, live_validate.py finalizes it.
Ported from the earlier pipeline's phases.py. The gates are no longer hardcoded: each phase's
conditions (metric + threshold) come from the task spec and are evaluated by revolve_common.metrics;
the check names are the metric names. The phase names below are the ported boundary rules' (the task
spec's phases must be these, in this order).

Phases (the episode, as replay.py builds it, or a policy's stream):
  approach  t = 0 -> the gripper close command
  grasp     the close command -> the first base/arm target motion after it (the pull)
  open      the pull start -> the gripper open command (pull + hold)
  release   the open command -> the end of the stream

Boundaries come only from the commanded actions, causally (each step sees its own action and
the previous one, never a later one), so the same rules work for the plan's 1 kHz actions and for
targets held at a policy's rate:
  approach -> grasp   first step whose gripper target is below the previous step's
  grasp -> open       first step after that whose base or arm target differs from the previous step's
  open -> release     first step after that whose gripper target is above the previous step's
The hold is the last hold_s before the release command; the commanded base/arm targets must be
static over it (the pull ended, `hold_duration`). The gates read the sim state:
  approach  TCP vs the plan's grasp pose (handoff tcp_world, row 0) at the phase end
  grasp     both fingers on the handle over the last grasp_check_s, finger openings vs the handle
            half-width at the phase end
  open      door / drawer min over the hold >= the task's target, handle slip in the TCP frame since
            the grasp end <= the limit, both fingers on the handle over the hold
  release   the global checks only
Global checks (peak body speed, penetration, robot contacts other than the fingers on the
handle) fail only the phase that holds their first violation (time stamps; contact messages
stamped by the step that made them). Untimed ones (gz/MuJoCo log, non-finite commands) are
"global, no regeneration".
"""
import math

import numpy as np

from revolve_common import metrics as M

PHASES = ("approach", "grasp", "open", "release")
CHANGE_TOL = 1e-9               # default; the task spec's boundaries.change_tol is passed in
# live check (metric) -> the offline check (verdict.json) it corresponds to (None: no offline counterpart)
OFFLINE = {"tcp_position_error_m": None, "tcp_rotation_error_deg": None, "grasp_width_error_m": "grasp_width",
           "finger_contacts": "both_fingers_on_handle", "handle_slip_m": "handle_slip",
           "joint_deg": "door_open_held", "joint_cm": "drawer_open_held",
           "targets_static_s": None, "peak_body_speed_mps": "peak_body_speed", "penetration_m": "penetration",
           "other_robot_contacts": "other_robot_contacts", "non_finite_commands": "non_finite_commands",
           "gz_log_problem_lines": "gz_and_mujoco_log_clean", "spawn_clearance_m": "spawn_in_collision"}


class Boundaries:
    """Causal phase boundaries from the commanded action stream (see the module doc)."""

    def __init__(self, phase="approach", change_tol=CHANGE_TOL):
        self.phase, self.tol = phase, change_tol
        self.prev = None                # (base/arm target, gripper target) of the previous step
        self.last_motion = None         # last step whose base/arm target changed

    def step(self, n, base, arm, grip, q=None):
        """Action of step n -> the phase that starts at step n, or None (q: unused here)."""
        tgt = np.concatenate([np.ravel(base), np.ravel(arm)])
        prev, self.prev = self.prev, (tgt, float(grip))
        if prev is None:
            return None
        moved = bool(np.any(np.abs(tgt - prev[0]) > self.tol))
        if moved:
            self.last_motion = n
        new = None
        if self.phase == "approach" and grip < prev[1] - self.tol:
            new = "grasp"
        elif self.phase == "grasp" and moved:
            new = "open"
        elif self.phase == "open" and grip > prev[1] + self.tol:
            new = "release"
        if new:
            self.phase = new
        return new


class PolicyBoundaries(Boundaries):
    """Boundaries of a policy rollout (policy_episode.py). A policy's targets move every frame and its
    gripper target jitters, so the plan rules above would fire on its first action. These use
    hold_check.py's windows instead (p: its closed_max_m, settle_s, moved_eps), still causally:
      approach -> grasp   first step whose gripper target is <= closed_max_m (closed) after one above it
      grasp -> open       settle_s after that, or the first step whose object joint q has moved more than
                          moved_eps from its value at the grasp start, whichever comes first
      open -> release     first step whose gripper target is above closed_max_m again (hold_check's release)
    p: the task spec's boundaries.policy with moved_eps in SI (rad or m)."""

    def __init__(self, p, dt, phase="approach"):
        super().__init__(phase)
        self.lim, self.settle_n, self.eps = p["closed_max_m"], int(round(p["settle_s"] / dt)), p["moved_eps"]
        self.g0 = None                  # (step, q) at the grasp start (a run restored in grasp: its first step)

    def step(self, n, base, arm, grip, q=None):
        prev, self.prev = self.prev, (None, float(grip))
        if prev is None:
            self.g0 = (n, q) if self.phase == "grasp" else None
            return None
        new = None
        if self.phase == "approach" and grip <= self.lim < prev[1]:
            new, self.g0 = "grasp", (n, q)
        elif self.phase == "grasp" and self.g0 is not None and (
                n - self.g0[0] >= self.settle_n or (q is not None and abs(q - self.g0[1]) > self.eps)):
            new = "open"
        elif self.phase == "open" and grip > self.lim >= prev[1]:
            new = "release"
        if new:
            self.phase = new
        return new


def check(name, ok, value, op, limit, unit="", t_fail=None, note=None):
    c = {"name": name, "pass": bool(ok), "value": value, "unit": unit, "threshold": {"op": op, "value": limit},
         "offline_check": OFFLINE.get(name), "t_fail": None if ok else t_fail}
    if note:
        c["note"] = note
    return c


def rot_deg(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))


def first_at_or_above(x, lim, s, e):
    """First index in [s, e) with x >= lim, else None."""
    i = np.flatnonzero(x[s:e] >= lim)
    return int(s + i[0]) if len(i) else None


# per-step series of the timed sanity metrics (phase_replay_system fills them) -> the metric's state key
SANITY_KEY = {"peak_body_speed_mps": "link_speeds", "penetration_m": "contact_depths",
              "other_robot_contacts": "other_robot_contacts"}


def violates(cond, value):
    """A condition's comparison fails for one sample's value."""
    return not M.CMP[cond.op](value, cond.threshold)


def from_result(r, t_fail=None, note=None, what=""):
    """revolve_common.metrics.evaluate result -> a phase check (name = metric)."""
    v = r["value"]
    v = round(v, 6) if isinstance(v, float) else v
    return check(r["metric"], r["pass_"], v, r["op"], r["threshold"], r["unit"] + (f", {what}" if what else ""),
                 t_fail=t_fail, note=r.get("note", note))


def global_checks(phase, s, e, first, per_step, conds, dt):
    """The timed sanity checks of one phase (steps [s, e)), from the task spec's sanity.timed conditions:
    each fails only in the phase holding its first violation over the whole run (first[metric] = step or
    None); the value is the metric over the phase."""
    out = []
    for c in conds:
        x = per_step[c.metric][s:e]
        v = M.value(c, {SANITY_KEY[c.metric]: x})
        k = first.get(c.metric)
        mine = k is not None and s <= k < e
        note = f"first violated at {round(k * dt, 3)} s (another phase)" if k is not None and not mine else None
        r = dict(metric=c.metric, op=c.op, threshold=c.threshold, unit=M.REGISTRY[c.metric].unit, pass_=not mine,
                 value=v)
        out.append(from_result(r, t_fail=round(k * dt, 3) if mine else None, note=note, what="sanity"))
    return out


def phase_result(name, s, e, checks, dt, state_file, extra=None):
    failed = [c for c in checks if not c["pass"]]
    t = [c["t_fail"] for c in failed if c["t_fail"] is not None]
    r = dict(status="fail" if failed else "pass", start_step=s, end_step=e, start_t=round(s * dt, 3),
             end_t=round(e * dt, 3), first_failure_t=min(t) if t else None,
             failed_checks=[c["name"] for c in failed], checks=checks, start_state=state_file)
    if extra:
        r.update(extra)
    return r


def episode_result(phases, global_untimed):
    """failed phase: the first failed phase; else global (no regeneration) if an untimed global check
    failed; else none (all passed)."""
    for p in PHASES:                     # a restored run has only the phases from its start on
        r = phases.get(p)
        if r is not None and r["status"] in ("fail", "not_reached"):
            return dict(failed_phase=p, regenerate=True, failed_checks=r["failed_checks"],
                        first_failure_t=r["first_failure_t"])
    bad = [c["name"] for c in global_untimed if not c["pass"]]
    if bad:
        return dict(failed_phase="global", regenerate=False, failed_checks=bad, first_failure_t=None)
    return dict(failed_phase=None, regenerate=False, failed_checks=[], first_failure_t=None)


def agreement(report, offline):
    """The 1 kHz rule: accepted <=> every phase (and the untimed globals) passes; a rejected
    episode fails at least one phase with a failed check whose offline counterpart is among the
    offline verdict's failed checks."""
    off_failed = {c["name"] for c in offline["checks"] if not c["pass"]}
    live_ok = report["result"]["failed_phase"] is None
    matched = []
    for p, r in report["phases"].items():
        for c in r.get("checks", []):
            if not c["pass"] and c["offline_check"] in off_failed:
                matched.append(f"{p}:{c['name']}")
    for c in report["global"]["checks"]:
        if not c["pass"] and c["offline_check"] in off_failed:
            matched.append(f"global:{c['name']}")
    live_names = {c["offline_check"] for r in report["phases"].values() for c in r.get("checks", [])
                  if not c["pass"]} | {c["offline_check"] for c in report["global"]["checks"] if not c["pass"]}
    ignore = {"gz_checks_all_pass", "all_checks_ran", "sim_budget"}
    if offline["verdict"] == "accepted":
        agree = live_ok
    else:
        agree = (not live_ok) and any(not m.startswith("global:") for m in matched)
    return dict(offline_verdict=offline["verdict"], offline_failed=sorted(off_failed), live_all_pass=live_ok,
                agree=agree, matched=matched,
                offline_not_failed_live=sorted(off_failed - live_names - ignore))


def summary_line(label, report):
    res = report["result"]
    ph = ", ".join(f"{p} {report['phases'][p]['status'].upper()}" for p in PHASES if p in report["phases"])
    if res["failed_phase"] is None:
        head = "ALL PASS"
    else:
        r = report["phases"].get(res["failed_phase"], {})
        bits = []
        for c in r.get("checks", report["global"]["checks"]):
            if not c["pass"]:
                bits.append(f"{c['name']} {c['value']} (needs {c['threshold']['op']} {c['threshold']['value']})")
        head = f"FAIL {res['failed_phase']}" + (" (no regeneration)" if not res["regenerate"] else "") + \
            f" at {res['first_failure_t']} s: " + "; ".join(bits)
    ag = report.get("agreement")
    tail = f" | offline {ag['offline_verdict']}, agree {'yes' if ag['agree'] else 'NO'}" if ag else ""
    return f"LIVE {label} feed {report['feed']}: {head} | {ph}{tail}"
