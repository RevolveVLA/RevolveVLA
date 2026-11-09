"""Accept/reject validation of one replay, shared by every task (replay.py verdict calls it).
Ported from the earlier pipeline's validation.py; the per-task validator.py files are
task_checks() here, driven by episode.json (hold part/unit and thresholds from the task spec).

  load(out)            episode.json and summary.json (kitchen_test.py) of a replay dir
  verdict(out)         every check -> verdict.json

Check order in verdict.json: spawn_in_collision, the task's checks (what it holds open, handle slip,
both fingers on the handle), then the generic ones below. The datagen / dataset steps read the
names, so keep them. Each check also has a `phase`, the episode phase it gates
(approach, grasp, open, release, or global = the whole episode).
"""
import json
import time
from pathlib import Path


def check(name, ok, value, op, limit, unit="", phase="global"):
    return {"name": name, "pass": bool(ok), "value": value, "unit": unit,
            "threshold": {"op": op, "value": limit}, "phase": phase}


def load(out):
    """episode.json, summary.json ({} before gz ran), the gz checks by name, the episode checks by kind."""
    out = Path(out)
    ep = json.loads((out / "episode.json").read_text())
    s = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    gz = {c["name"]: c for c in s.get("checks", [])}
    res = {r["kind"]: r for r in s.get("results", [])}
    return ep, s, gz, res


def spawn_check(ep):
    """The spawn state against the scene (the clearance precheck geometry). MuJoCo would push a
    robot spawned in penetration out at t = 0, so such a spawn is rejected before gz."""
    c, lim = ep["clearance"]["settle"], ep["thresholds"]["spawn_clearance_min_m"]
    return check("spawn_in_collision", c["min_m"] >= lim, round(c["min_m"] * 1000, 1),
                 ">=", lim * 1000, f"mm min spawn clearance, {c['link']} / {c['obstacle']}, checked before gz", "approach")


def task_checks(ep, res):
    """<part>_open_held, handle_slip, both_fingers_on_handle (verdict.json order; were validator.checks of
    open_fridge_door and open_drawer) from episode.json and the gz episode results by kind."""
    H, th = ep["hold"], ep["thresholds"]
    part, unit = H["part"], H["unit"]
    g, d = res.get("grasped", {}), res.get(H["check"], {})
    v, lim = d.get(f"{part}_min_{unit}"), th[f"{part}_open_min_{unit}"]
    out = [check(f"{part}_open_held", v is not None and v >= lim, v, ">=", lim,
                 f"{unit}, min over the {th['hold_s']} s hold", "open")]
    v, lim = d.get("slip_mm"), th["handle_slip_max_m"] * 1000
    out.append(check("handle_slip", v is not None and v <= lim, v, "<=", lim, "mm, grasp check to the end of the hold",
                     "open"))
    v = [g.get("contact_both_fingers"), d.get("contact_both_fingers")]
    out.append(check("both_fingers_on_handle", all(x is True for x in v), v, "==", [True, True], "grasp check, hold",
                     "grasp"))
    return out


def generic_checks(ep, s, gz, res):
    """The checks every task shares: grasp width, physics, contacts, controller, budget, gz log."""
    th = ep["thresholds"]

    def gzc(name, i=None):
        c = gz.get(name, {})
        v = c.get("value")
        return c.get("pass", False), (v[i] if i is not None and isinstance(v, list) and v else v)

    out = []
    v, half = res.get("grasped", {}).get("fingers_mm"), th["grasp_half_width_m"] * 1000
    tol = th["grasp_width_tol_m"] * 1000
    out.append(check("grasp_width", v is not None and all(abs(f - half) < tol for f in v), v, "within",
                     [half - tol, half + tol], "mm per finger at the grasp check", "grasp"))
    ok, v = gzc("physics: peak body speed (m/s)", 0)
    out.append(check("peak_body_speed", ok, v, "<", th["peak_body_speed_max_mps"], "m/s"))
    ok, v = gzc("physics: max contact penetration (cm)", 0)
    out.append(check("penetration", ok, v, "<", th["penetration_max_m"] * 100, "cm"))
    ok, v = gzc("robot contacts other than fingers on the grasped object")
    out.append(check("other_robot_contacts", ok, len(s.get("other_robot_contacts", [])) if s else None, "==", 0,
                     "contacts (first 50 kept in summary.json)"))
    ok, v = gzc("controller: non-finite joint commands (zeroed)")
    out.append(check("non_finite_commands", ok, len(v) if isinstance(v, list) else v, "==", 0, "commands"))
    t_last = max((r["t"] for r in s.get("results", []) if "t" in r), default=None)
    out.append(check("sim_budget", res.get("budget", {}).get("pass_", False), t_last, "<=", ep["budget_s"],
                     "s sim, end of the last check"))
    ok, v = gzc("all planned checks ran")
    out.append(check("all_checks_ran", ok, v, ">=", len(ep["checks"]), "checks"))
    ok, v = gzc("load: no [Err] / NaN lines in the gz log, no MUJOCO_LOG")
    out.append(check("gz_and_mujoco_log_clean", ok, len(v) if isinstance(v, list) else v, "==", 0, "problem lines"))
    out.append(check("gz_checks_all_pass", s.get("pass_all") is True, s.get("pass_all"), "==", True,
                     "every check kitchen_test.py printed (guards anything not listed above)"))
    return out


def verdict(out, t_start=None):
    """<out>/verdict.json from episode.json + summary.json. Exit code 0 = accepted."""
    out = Path(out)
    ep, s, gz, res = load(out)
    H = ep["hold"]
    part, unit = H["part"], H["unit"]
    d = res.get(H["check"], {})
    spawn = spawn_check(ep)
    checks = [spawn]
    if spawn["pass"]:                  # else rejected before gz (replay.py actions): no gz check ran
        checks += task_checks(ep, res) + generic_checks(ep, s, gz, res)
    failed = [c for c in checks if not c["pass"]]
    ex = s.get("extra", {})
    rep = ex.get("replay", {})
    out_v = dict(
        plan_run=ep["plan_run"], attempt_id=ep["attempt_id"], grasp_id=ep["grasp_id"],
        verdict="rejected" if failed else "accepted",
        reasons=[f"{c['name']}: {c['value']} (needs {c['threshold']['op']} {c['threshold']['value']} {c['unit']})"
                 for c in failed],
        checks=checks,
        metrics={f"{part}_max_{unit}": (ex.get("episode", {}).get(f"{part}_max_{unit}") or [None])[0],
                 f"{part}_hold_min_{unit}": d.get(f"{part}_min_{unit}"), "handle_slip_mm": d.get("slip_mm"),
                 "peak_body_speed": s.get("peak_speed"), "penetration": s.get("penetration"),
                 f"planned_{part}_final_{unit}": ep[f"planned_{part}_final_{unit}"], "pull_s": ep["timing"]["pull_s"],
                 "sim_s": rep.get("sim_s"), "loop_wall_s": rep.get("loop_wall_s"), "rtf": rep.get("rtf"),
                 "wall_s": round(time.time() - float(t_start), 1) if t_start else None,
                 "real_time_factor_cap": rep.get("real_time_factor_cap")},
        timing=ep["timing"], clearance=ep["clearance"],
        files=sorted(p.name for p in out.iterdir()))
    (out / "verdict.json").write_text(json.dumps(out_v, indent=1))
    print(f"VERDICT {out_v['verdict']} {ep['plan_run']}/{ep['attempt_id']}: "
          + ("; ".join(out_v["reasons"]) if failed else f"{len(checks)}/{len(checks)} checks pass"))
    return 1 if failed else 0
