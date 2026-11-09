"""Replay of one AutoMoMa plan in gz: the action sequence before the run, the verdict after it.
Numpy only (no torch); runs in revolve-simulator (cli.py replay). Ported from
the earlier pipeline's replay.py: the task, robot and scene come from specs/ (spec.py), every
threshold from the task spec.

  replay.py actions <plan run dir> <attempt id> <out dir> [<task name>]
      handoff.npz row -> <out>/actions.npz (mobile_panda.ActionSeq, one whole-body Action per
      1 ms step) and <out>/episode.json (events, check windows, thresholds, timing, clearance,
      the task object's joint, handle and bag topic). The task is the plan's (handoff.json) unless named.
      A spawn state in collision is rejected here: verdict.json (spawn_in_collision), exit 1, no gz.
  replay.py verdict <out dir> [start wall time, s]
      <out>/summary.json (kitchen_test.py) -> <out>/verdict.json: accepted or rejected, every
      check with its value and threshold, the failed checks as reasons. Exit code 0 = accepted.
      The checks: validation.py (shared and the task's, from its spec).

Episode built from the gz skill building blocks (Plan, Scenario, mobile_panda_skills.close):
  approach  spawn at the plan's first base moved standoff_m back along the hand axis, the arm at
            the plan's first arm config, gripper open (the replay system resets the robot to the
            first action at t = 0); settle settle_s; base in to the plan's first base. The arm is
            fixed while the base moves in, so the TCP goes straight in along the hand axis and
            ends exactly at the plan's row 0.
  grasp     close 1.5 s + settle 0.5 s (rows 0-3 of the plan are this closing), grasp check 0.5 s
  pull      the 32 trajopt waypoints (rows 4-35) at uniform dt, linear in between
  hold      hold_s at the last waypoint, handle grasped (door or drawer open, slip check)
  release   open 1.5 s, base back out retreat_m along the hand axis, hold 0.5 s

With the round's `timing` block (env ROUND_TIMING, core.round_key_env) the replay takes settle_s,
close_s, grip_settle_s, pull_start_s and hold_s from it, on the 10 ms plan grid (timed_grasp):
no settle event at 0 s, the pull starts pull_start_s after the close command (the close may run
on into the pull), and the grasp check is the first 10 ms of the pull ("pull", then "pull on").
The plans repeat their first pull waypoint, a still dt (8/31 s or more) after the pull start, so
the timed pull drops the repeats and keeps the waypoint dt.
With a `pull` block too (env ROUND_PULL) the drop also takes the leading waypoints within
repeat_tol of the first: some plans repeat it up to float32 noise (2.4e-7), which the exact drop
keeps as one still dt (DX).
Without the block the replay is master's.
"""
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import mobile_panda as mp
import mobile_panda_skills as sk
import panda_plan as pp
import sdf
import spec
from validation import spawn_check, verdict

SUBTASKS = ("approach", "grasp", "pull", "hold", "release")
SUBTASK_START = {"settle": "approach", "grip close": "grasp", "pull": "pull",
                 "check door_open": "hold", "check drawer_open": "hold", "release": "release"}
CLEARANCE_EVERY = 5          # plan rows (10 ms) between clearance samples
GRASP_CHECK_S = 0.5          # grasp check hold before the pull [s]
# With a timing block the grasp check is the first plan row of the pull. gz takes the handle's slip
# reference at the window end, so it must come before the 5-9 mm in-hand transient of the pull start.
TIMED_GRASP_CHECK_S = mp.PLAN_DT
NO_CHASSIS = {"base_link", "panda_link0", "panda_link1", "panda_link2"}   # on the chassis


def hand_axis_xy(q, base):
    z = mp.fk(q, base)[0][:3, 2]
    return np.array([*(z[:2] / np.linalg.norm(z[:2])), 0.0])


def hold_rows(P, rows, label):
    """mp.Plan.hold for an exact number of plan rows, and no event for 0 rows (Plan.hold pushes 2)."""
    if rows > 0:
        P.events.append((P.t, label, {}))
        P.rows += [(P.b.copy(), P.q.copy(), P.g, P.f) for _ in range(rows)]


def timed_grasp(P, b0, rp, t):
    """Settle, approach and close of a timing block t: settle_s, base in over approach_s, the
    gripper closes over close_s, and the pull starts pull_start_s after the close command. A gap
    before the pull holds: 'grip settle' up to grip_settle_s, then 'grip wait'. Returns the
    gripper targets of the close rows after the pull start (pull_start_s < close_s): the pull
    rows carry them."""
    n = lambda s: int(round(s / mp.PLAN_DT))
    hold_rows(P, n(t["settle_s"]), "settle")
    P.drive(b0, rp["approach_s"], label="approach in")
    k = len(P.rows) + n(t["pull_start_s"])
    P.gripper(mp.GRIP_CLOSED, t["close_s"], label="grip close")
    rest = [r[2] for r in P.rows[k:]]
    del P.rows[k:]
    hold_rows(P, min(n(t["grip_settle_s"]), k - len(P.rows)), "grip settle")
    hold_rows(P, k - len(P.rows), "grip wait")
    return rest


def build(run, attempt, task, robot, timing=None, pull=None):
    """Scenario (Plan + check windows) and the episode facts for one planned attempt; task: spec.legacy_task;
    robot: sdf.Model of the gz robot (velocity limits); timing, pull: the round's blocks or None
    (pull acts only with timing)."""
    h = np.load(run / "handoff.npz")
    meta = json.loads((run / "handoff.json").read_text())
    k = int(np.flatnonzero(h["attempt_id"] == attempt)[0])
    if not h["success"][k]:
        raise SystemExit(f"attempt {attempt} did not plan (success false): nothing to replay")
    det = next(d for d in meta["attempts_detail"] if d["attempt_id"] == attempt)
    rp = tm = task["replay"]                               # pull_slowdown, pull_min_s: the task's replay block
    obj, part, unit, si, from_si = spec.hold(spec.task(task["name"]))
    W = h["traj_robot_world"][k, 4:, :10].copy()          # 32 waypoints: base x, y, yaw, arm 1-7
    clipped = float(np.abs(np.clip(W[:, 3:], pp.LOWER, pp.UPPER) - W[:, 3:]).max())
    W[:, 3:] = np.clip(W[:, 3:], pp.LOWER, pp.UPPER)       # cuRobo rounds the limits outwards
    yaw_offset = mp.wrap(W[0, 2]) - W[0, 2]                # 2 pi k: spawn yaw in (-pi, pi]
    W[:, 2] += yaw_offset
    b0, q0 = W[0, :3], W[0, 3:]

    sc = sk.Scenario(f"automoma_{run.name}_{attempt}")
    # spawn: staging base, arm at q0, gripper open (Scenario's own Plan starts at BASE0, READY)
    sc.P = P = mp.Plan(b0 - rp["standoff_m"] * hand_axis_xy(q0, b0), q0)
    pull_grip = []
    if timing is None:
        P.hold(rp["settle_s"], "settle")
        P.drive(b0, rp["approach_s"], label="approach in")
        sk.close(P)                                        # "grip close" 1.5 s, "grip settle" 0.5 s
        sc.check("grasped", GRASP_CHECK_S, obj=obj)
    else:
        pull_grip = timed_grasp(P, b0, rp, timing)
    T = max(tm["pull_min_s"], tm["pull_slowdown"] * det["duration_hint_s"])
    if timing is not None:      # the plans repeat their first waypoint: drop that still dt at the pull start
        n0 = int(np.argmax(np.abs(W[1:] - W[0]).max(axis=1) > (pull["repeat_tol"] if pull else 0)))
        W, T = W[n0:], T * (len(W) - 1 - n0) / (len(W) - 1)
    t_wp = np.linspace(0.0, T, len(W))
    P.events.append((P.t, "pull", dict(duration_s=T, dt_waypoint_s=T / (len(W) - 1))))
    # timed: the pull row holds the first waypoint, so the 1 ms upsampling moves nothing before it
    for j, ts in enumerate(np.arange(0 if timing else 1, round(T / mp.PLAN_DT) + 1) * mp.PLAN_DT):
        if timing is not None and j == round(TIMED_GRASP_CHECK_S / mp.PLAN_DT):
            sc.checks.append((len(P.events) - 1, "grasped", dict(obj=obj)))     # window: the "pull" event
            P.events.append((P.t, "pull on", {}))
        x = np.array([np.interp(ts, t_wp, W[:, i]) for i in range(W.shape[1])])
        P.b, P.q = x[:3], x[3:]
        P.rows.append((P.b.copy(), P.q.copy(), pull_grip[j] if j < len(pull_grip) else P.g, P.f))
    sc.check(rp["hold_check"], rp["hold_s"], obj=obj, want=si(rp[f"{part}_open_min_{unit}"]))
    P.gripper(mp.GRIP_OPEN, 1.5, label="release")
    P.drive(P.b - rp["retreat_m"] * hand_axis_xy(P.q, P.b), rp["retreat_s"], label="retreat")
    P.hold(0.5, "done")
    sc.info = dict(budget=rp["budget_s"])

    vlim = np.array([mp.BASE_V_MAX, mp.BASE_V_MAX, mp.BASE_W_MAX] +
                    [robot.joints[f"panda_joint{i}"]["velocity"] for i in range(1, 8)])
    seg = np.abs(np.diff(W, axis=0)) / (T / (len(W) - 1))
    facts = dict(plan_run=run.name, attempt_id=attempt, grasp_id=int(h["grasp_id"][k]),
                 **{key: det[key] for key in ("grasp_x", "grasp_y", "grasp_z") if key in det},
                 **{f"planned_{part}_final_{unit}": from_si(h["door_angle"][k, -1])},
                 yaw_offset_rad=yaw_offset, arm_clip_max_rad=clipped,
                 timing=dict(rule=f"uniform dt over the 32 waypoints, linear in between; pull = max("
                                  f"{tm['pull_min_s']} s, {tm['pull_slowdown']} x duration_hint_s)"
                                  + ("; repeats of the first waypoint dropped" if timing else ""),
                             duration_hint_s=det["duration_hint_s"], pull_s=T, dt_waypoint_s=T / (len(W) - 1),
                             peak_fraction_of_velocity_limits=float((seg / vlim).max())))
    return sc, facts, float(h["door_angle"][k, -1])


def clearance(P, door_final, task, scene, robot):
    """Min distance [m] per phase (settle = the spawn state, approach in, retreat, done) from the
    robot's collision shapes (capsules sampled as spheres, boxes as grid points) to the scene
    boxes (the fridge and the task object box by box, the task joint closed during the approach
    and at its planned final position after the pull; other models as their AABBs; walls) and,
    for the arm, to its own chassis.
    The pull itself is the planner's (cuRobo spheres, T2 clearance check)."""
    import mobile_panda_smoke_system as smoke              # gz image only (gz.* imports)
    shapes = {}
    for ln, cols in smoke.collisions(mp.MODEL).items():
        pts = [smoke.sample_points(kind, T, dims) for kind, T, dims in cols]
        shapes[ln] = (np.vstack([p for p, _ in pts]), np.concatenate([r for _, r in pts]))
    chassis = next((T, np.array(d)) for kind, T, d in smoke.collisions(mp.MODEL)["base_link"] if kind == "box")

    def boxes(door):
        out = []
        for name, uri, Tm, inline in sdf.world_entities(spec.path(scene["world"])):
            if name in scene["skip"]:
                continue
            if inline is not None or name in scene["detailed"] or name == task["object"]:
                m = inline or sdf.Model(spec.model_dir(scene["assets"], uri) / "model.sdf")
                assert not m.meshes, f"{name}: mesh collisions {m.meshes} (box-only scene, unseen)"
                q = {task["joint"]: door} if name == task["object"] else {}
                out += [(f"{name}::{c}", Tm @ m.pose(ln, q) @ Tc, np.array(s))
                        for ln, bs in m.boxes.items() for c, Tc, s in bs]
            else:
                lo, hi = np.array(json.loads((spec.model_dir(scene["assets"], uri) / "manifest.json")
                                             .read_text())["aabb_model_frame"])
                out.append((f"{name}::aabb", Tm @ sdf.tf((lo + hi) / 2), hi - lo))
        return out

    def sd_box(p, T, size):
        q = np.abs((p - T[:3, 3]) @ T[:3, :3]) - size / 2
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)

    starts = [(round(t / mp.PLAN_DT), lab) for t, lab, _ in P.events]
    phase_of = {}
    for i, (k0, lab) in enumerate(starts):
        k1 = starts[i + 1][0] if i + 1 < len(starts) else len(P.rows)
        for k in range(k0, k1):
            phase_of[k] = lab
    # not "release": the closed fingers sit on the handle there by design
    keep = {"settle", "approach in", "retreat", "done"}
    world = {False: boxes(0.0), True: boxes(door_final)}
    res, after = {}, False
    for k in range(0, len(P.rows), CLEARANCE_EVERY):
        lab = phase_of[k] if k else "settle"                # row 0 is the spawn, also without a settle event
        after |= lab == "release"
        if lab not in keep:
            continue
        b, q, g, _ = P.rows[k]
        qd = dict(base_x=b[0], base_y=b[1], base_yaw=b[2], panda_finger_joint_left=g, panda_finger_joint_right=g,
                  **{f"panda_joint{i + 1}": v for i, v in enumerate(q)})
        Tc = robot.pose("base_link", qd) @ chassis[0]
        for ln, (pts, rad) in shapes.items():
            Tl = robot.pose(ln, qd)
            p = pts @ Tl[:3, :3].T + Tl[:3, 3]
            cands = world[after] + ([] if ln in NO_CHASSIS else [("mobile_panda::chassis", Tc, chassis[1])])
            for name, Tb, size in cands:
                d = float((sd_box(p, Tb, size) - rad).min())
                if lab not in res or d < res[lab][0]:
                    res[lab] = [d, ln, name, round(k * mp.PLAN_DT, 2)]
    return {lab: dict(min_m=round(v[0], 4), link=v[1], obstacle=v[2], t=v[3]) for lab, v in res.items()}


def actions(run, attempt, out, task_name=None):
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = json.loads((run / "handoff.json").read_text())["plugins"]
    r = spec.task(task_name or names["task"])
    if spec.RENAMED.get(names["scene"], names["scene"]) != r.scene.name:
        raise SystemExit(f"plan scene {names['scene']} != task {r.task.name}'s scene {r.scene.name}")
    task, scene = spec.legacy_task(r), spec.legacy_scene(r)
    robot = sdf.Model(os.environ.get("SIM_ROBOT_MODEL") or spec.path(r.robot.camera_model))
    robot.zero = functools.lru_cache(maxsize=None)(robot.zero)
    timing = json.loads(os.environ["ROUND_TIMING"]) if os.environ.get("ROUND_TIMING") else None
    pull = json.loads(os.environ["ROUND_PULL"]) if os.environ.get("ROUND_PULL") else None
    if timing:                                     # the block's settle and hold replace the task plugin's
        task = dict(task, replay=dict(task["replay"], settle_s=timing["settle_s"], hold_s=timing["hold_s"]))
    sc, facts, door_final = build(run, int(attempt), task, robot, timing, pull)
    seq = sc.P.actions()
    seq.save(out / "actions.npz")
    sub, cur = [], SUBTASKS[0]
    for _, lab, _ in sc.P.events:
        cur = SUBTASK_START.get(lab, cur)
        sub.append(cur)
    rp = task["replay"]
    obj, part, unit = spec.hold(r)[:3]
    man = json.loads((spec.path(task["gz_model"]) / "manifest.json").read_text())
    hd = next(x for x in man["handles"] if x["name"] == task["handle"])
    half = spec.grasp_half_width(r, hd["size"])    # per-finger opening the gz grasp check expects
    ep = dict(facts, control_dt=mp.CONTROL_DT, steps=len(seq), duration_s=sc.P.t, budget_s=sc.info["budget"],
              contact_rate_hz=spec.SIM["contact_rate_hz"],
              task=task["name"], object_joint=f"{task['object']}::{task['joint']}",
              object_handle=f"{task['object']}::{hd['link']}::{task['handle']}", joint_topic=task["joint_topic"],
              prompt=task["prompt"],
              hold=dict(check=rp["hold_check"], obj=obj, part=part, unit=unit),
              thresholds=dict(spec.legacy_thresholds(r), grasp_half_width_m=half),
              approach=dict(standoff_m=rp["standoff_m"], settle_s=rp["settle_s"], approach_s=rp["approach_s"],
                            retreat_m=rp["retreat_m"]),
              events=[[round(t, 6), lab, info] for t, lab, info in sc.P.events], event_subtask=sub,
              subtasks=list(SUBTASKS), checks=[[i, kind, info] for i, kind, info in sc.checks],
              **({"round_timing": timing} if timing else {}))
    t0 = time.time()
    ep["clearance"] = clearance(sc.P, door_final, task, scene, robot)
    ep["clearance_s"] = round(time.time() - t0, 1)
    (out / "episode.json").write_text(json.dumps(ep, indent=1, default=float))
    worst = min(ep["clearance"].items(), key=lambda kv: kv[1]["min_m"])
    print(f"attempt {attempt}: {len(seq)} steps ({sc.P.t:.1f} s sim), pull {facts['timing']['pull_s']:.1f} s "
          f"(peak {facts['timing']['peak_fraction_of_velocity_limits']:.2f} of the velocity limits), "
          f"planned {part} {facts[f'planned_{part}_final_{unit}']:.1f} {unit}, "
          f"min clearance {worst[1]['min_m'] * 1000:.1f} mm "
          f"({worst[0]}: {worst[1]['link']} / {worst[1]['obstacle']})")
    return 0 if spawn_check(ep)["pass"] else verdict(out)    # spawn in collision: rejected, gz not run


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    sys.exit({"actions": actions, "verdict": verdict}[cmd](*args))
