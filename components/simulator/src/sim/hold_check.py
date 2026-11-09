"""Closed-loop success: the expert episode check (replay.py verdict: the object joint at or above the threshold over
the whole hold, handle slip in the TCP frame from the grasp check to the hold end, the finger openings at the grasp
check, both fingers on the handle during the grasp check and the hold) on windows found without a plan. The
expert's windows sit on its plan's gripper events; here they sit on the gripper target (the applied action):

  closed     gripper target <= grip_closed_max_m (the plan commands GRIP_CLOSED, 0)
  release r  the first step of a run where the target is not closed (the plan's "release" = the hold end); the
             last step when the episode ends closed
  grasp g    grasp_settle_s after the start of the closed run that ends at r (the plan: close ramp end + grip
             settle 0.5 s + grasp check 0.5 s), or the last step before the joint moved joint_moved_eps from its
             value at that start (a pull that began earlier), whichever is first
  hold       steps r - hold_s .. r

Task parameters: the task plugin's `replay` (threshold, hold_s, handle_slip_max_m) and `closed_loop_success`
blocks; the simulator plugin's grasp_width_tol_m. Numpy only: closed_loop_system calls it at every release, and
an offline tool can call it on a state_log.npz.
"""
import numpy as np


def part(ok, value):
    return {"pass": bool(ok), "value": value}


def evaluate(r, q, grip, fingers, handle, touched, p, dt):
    """Hold check at release step r. Arrays per 1 ms step (row n = the state at the start of step n), valid up to
    r: q object joint [rad or m], grip gripper target [m], fingers (n, 2) finger openings [m], handle (n, 3) the
    handle link's origin in the TCP frame [m]. touched(t0, t1): both fingers touched the handle in [t0, t1] (sim s).
    p: threshold, hold_s, slip_max_m, half_width_m, width_tol_m, closed_max_m, settle_s, window_s, moved_eps
    (SI). -> dict(pass, reached, release_step, grasp_step, slip_mm, hold_min, parts)."""
    hold_n, settle_n, win_n = (int(round(p[k] / dt)) for k in ("hold_s", "settle_s", "window_s"))
    closed = np.asarray(grip[: r + 1]) <= p["closed_max_m"]
    end = r if closed[r] else r - 1                    # last closed step of the run
    parts = {}
    res = {"pass": False, "reached": bool(np.max(q[: r + 1]) >= p["threshold"]), "release_step": int(r),
           "grasp_step": None, "slip_mm": None, "hold_min": None, "parts": parts}
    if end < 0 or not closed[end]:
        parts["grasp_ref"] = part(False, "gripper not closed before the release")
        return res
    opened = np.flatnonzero(~closed[: end + 1])
    s0 = int(opened[-1]) + 1 if len(opened) else 0
    moved = np.flatnonzero(np.abs(np.asarray(q[s0: end + 1]) - q[s0]) > p["moved_eps"])
    g = min(s0 + settle_n, end, s0 + int(moved[0]) - 1 if len(moved) else end)
    g = max(g, s0)
    res["grasp_step"] = g
    parts["grasp_ref"] = part(True, dict(closed_from=s0, grasp=g, closed_s=round((g - s0) * dt, 3)))
    k0 = r - hold_n
    if k0 < 0:
        parts["held"] = part(False, "hold window before the episode start")
        return res
    hmin = float(np.min(q[k0: r + 1]))
    res["hold_min"] = hmin
    parts["held"] = part(hmin >= p["threshold"], hmin)
    f = np.asarray(fingers[g], float)
    parts["grasp_width"] = part(np.all(np.abs(f - p["half_width_m"]) < p["width_tol_m"]), (f * 1000).round(2).tolist())
    parts["fingers_grasp"] = part(bool(touched((g - win_n) * dt, g * dt)), [(g - win_n) * dt, g * dt])
    parts["fingers_hold"] = part(bool(touched(k0 * dt, r * dt)), [k0 * dt, r * dt])
    slip = float(np.linalg.norm(np.asarray(handle[r]) - np.asarray(handle[g])))
    res["slip_mm"] = round(slip * 1000, 2)
    parts["slip"] = part(slip <= p["slip_max_m"], res["slip_mm"])
    res["pass"] = all(v["pass"] for v in parts.values())
    return res
