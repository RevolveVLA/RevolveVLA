"""Recovery, gz side (numpy only; runs in revolve-simulator through components/simulator/launch.py with
cli=this file). The planner side is plan_from_state.py (revolve-recovery); recover.py drives both.
Ported from the earlier pipeline's regenerate.py + gz_regen.sh: the task, robot and scene from
specs/ (the simulator's spec.py), the gz run, phase gates and acceptance through the simulator's
cli.phase_run + live_validate.finalize (replay mode: the offline verdict of the regenerated part AND
every task.yaml phase from the restarted one on), restored from the saved state in step 0.

  regen_sim.py candidate --plan P --attempt A --out D [--cameras HZ] [--no-bag]
      one planned attempt of a plan_from_state run P: the regenerated part's actions.npz + episode.json
      (starting at P/start_state.npz), its clearance precheck over every planned row, then (precheck
      passed) the gz run from the restored state, and with --cameras the training cameras + the MCAP bag
      (bag/, bag_check.json) as the simulator's replay. Exit 0 = accepted, 1 = rejected in gz,
      2 = precheck failed (no gz), 3 = setup, 4 = bag check failed.
  regen_sim.py perturb --source EP --out D --joint X [--settle-s 0.5]
      test support: a "policy-like" open start that no plan produced. EP's run restored at its
      open_start, the pull fed until the task joint reaches X deg (cm for a drawer), then the targets
      held settle_s with the gripper closed, then the gripper opened. Its release_start snapshot (door
      partly open, the handle held where the pull left it) becomes D/rollout/phase_states/open_start.npz
      with the held (closed) command and the handle-in-TCP grasp_rel measured in that state; D/rollout
      gets an episode.json (policy_rollout) and a phase_report.json failing open, as a rollout's.

The regenerated part, as replay.py builds a whole episode (same building blocks and timing):
  open      (restart at the pull start, the handle grasped) grasp check 0.5 s at the saved
            commanded state, so the offline grasp width / both fingers / slip reference cover the
            regenerated part; then the planned pull, hold, release, retreat
  approach  (restart at the spawn, or at the grasp start for a grasp failure) the MotionGen reach from
  / grasp   the saved state to the staging pose of the new grasp (its pull start moved standoff_m back
            along the hand axis), then approach in, grip close, grasp check, pull, hold, release, retreat
"""
import argparse
import functools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

import cli
import live_validate
import replay as rp_
import robots
import sdf
import skills as sk
import spec
from robots.base import CONTROL_DT, PLAN_DT, Plan, tf

CLEAR_EVERY = 5                 # plan rows (50 ms) between clearance samples
GENERATOR = spec.ROOT / "components" / "datagen" / "src" / "datagen" / "generator.yaml"


def regen_settings():
    return yaml.safe_load(GENERATOR.read_text())["regenerate"]


def load_snapshot(path):
    d = np.load(path, allow_pickle=True)
    s = {k: d[k] for k in d.files}
    names = [str(x) for x in s["joint_names"]]
    scope = f"{robots.current().model}::"
    s["robot"] = {n.split("::")[1]: float(v) for n, v in zip(names, s["joint_pos"]) if n.startswith(scope)}
    s["joints"] = {n: float(v) for n, v in zip(names, s["joint_pos"])}
    return s


def unwrap_to(yaw, ref):
    """yaw + 2 pi k closest to ref."""
    return yaw + 2 * math.pi * round((ref - yaw) / (2 * math.pi))


class Builder:
    """The regenerated part as a Plan with check windows (sk.Scenario without its settle at BASE0)."""

    def __init__(self, b, q, g, f):
        self.P = Plan(robots.current(), b, q, g, f)
        self.checks = []

    def check(self, kind, dur, **info):
        self.P.hold(dur, f"check {kind}", info)
        self.checks.append((len(self.P.events) - 1, kind, info))


def build(plan, attempt, r, task):
    """Builder + episode facts for one attempt of a plan_from_state run (task: spec.legacy_task(r))."""
    h = np.load(plan / "handoff.npz")
    meta = json.loads((plan / "handoff.json").read_text())
    rg = meta["regenerate"]
    k = int(np.flatnonzero(h["attempt_id"] == attempt)[0])
    if not h["success"][k]:
        raise SystemExit(f"attempt {attempt} did not plan: nothing to build")
    det = next(d for d in meta["attempts_detail"] if d["attempt_id"] == attempt)
    snap = load_snapshot(plan / "start_state.npz")
    rp = tm = task["replay"]                               # pull_slowdown, pull_min_s: the task's replay block
    obj, part, unit, si, from_si = spec.hold(r)
    rb = robots.current()
    nq = 3 + rb.dof                                        # base + arm columns
    W = h["traj_robot_world"][k, 4:, :nq].copy()
    clipped = float(np.abs(np.clip(W[:, 3:], rb.lower, rb.upper) - W[:, 3:]).max())
    W[:, 3:] = np.clip(W[:, 3:], rb.lower, rb.upper)
    b_s = np.array(snap["action_base"], float)
    q_s = np.array(snap["action_arm"], float)
    if rg["mode"] == "pull":
        yaw_offset = unwrap_to(W[0, 2], b_s[2]) - W[0, 2]
        W[:, 2] += yaw_offset
        B = Builder(b_s, q_s, float(snap["action_grip"]), float(snap["action_grip_force"]))
        B.check("grasped", 0.5, obj=obj)
        start_jump = np.abs(W[0] - np.r_[b_s, q_s])
    else:                                       # reach: reach, approach in, close, grasp check
        R = h["reach_world"][k][: int(h["reach_len"][k])].copy()      # (n, nq) at reach_dt
        yaw_offset = unwrap_to(R[0, 2], b_s[2]) - R[0, 2]
        R[:, 2] += yaw_offset
        W[:, 2] = unwrap_to(W[0, 2], R[-1, 2]) - W[0, 2] + W[:, 2]
        R[:, 3:] = np.clip(R[:, 3:], rb.lower, rb.upper)
        B = Builder(b_s, q_s, rb.gripper.open_m, float(snap["action_grip_force"]))
        P = B.P
        dt_r = float(h["reach_dt"][k]) * float(rg["reach_slowdown"])
        t_r = np.arange(len(R)) * dt_r
        P.events.append((P.t, "reach", dict(duration_s=float(t_r[-1]), planner_dt_s=float(h["reach_dt"][k]))))
        for ts in np.arange(1, round(t_r[-1] / PLAN_DT) + 1) * PLAN_DT:
            x = np.array([np.interp(ts, t_r, R[:, i]) for i in range(nq)])
            P.b, P.q = x[:3], x[3:]
            P.rows.append((P.b.copy(), P.q.copy(), P.g, P.f))
        P.hold(0.3, "reach settle")
        b0 = W[0, :3]
        P.drive(b0, rp["approach_s"], label="approach in")
        sk.close(P)
        B.check("grasped", 0.5, obj=obj)
        start_jump = np.abs(R[0] - np.r_[b_s, q_s])
    P = B.P
    T = max(tm["pull_min_s"], tm["pull_slowdown"] * det["duration_hint_s"])
    t_wp = np.linspace(0.0, T, len(W))
    P.events.append((P.t, "pull", dict(duration_s=T, dt_waypoint_s=T / (len(W) - 1))))
    for ts in np.arange(1, round(T / PLAN_DT) + 1) * PLAN_DT:
        x = np.array([np.interp(ts, t_wp, W[:, i]) for i in range(W.shape[1])])
        P.b, P.q = x[:3], x[3:]
        P.rows.append((P.b.copy(), P.q.copy(), P.g, P.f))
    B.check(rp["hold_check"], rp["hold_s"], obj=obj, want=si(rp[f"{part}_open_min_{unit}"]))
    P.gripper(rb.gripper.open_m, 1.5, label="release")
    P.drive(P.b - rp["retreat_m"] * rp_.hand_axis_xy(P.q, P.b), rp["retreat_s"], label="retreat")
    P.hold(0.5, "done")
    facts = dict(plan_run=plan.name, attempt_id=attempt, grasp_id=int(h["grasp_id"][k]),
                 **{key: det[key] for key in ("grasp_x", "grasp_y", "grasp_z") if key in det},
                 **{f"planned_{part}_final_{unit}": from_si(h["door_angle"][k, -1])},
                 yaw_offset_rad=yaw_offset, arm_clip_max_rad=clipped,
                 start_jump_max=float(start_jump.max()),
                 timing=dict(rule=f"uniform dt over the 32 waypoints, linear in between; pull = max("
                                  f"{tm['pull_min_s']} s, {tm['pull_slowdown']} x duration_hint_s)",
                             duration_hint_s=det["duration_hint_s"], pull_s=T, dt_waypoint_s=T / (len(W) - 1)))
    return B, facts, h["door_angle"][k], rg, snap


class Clearance:
    """Min distance of the gz robot's collision shapes (as replay.clearance samples them) to the scene
    boxes, with the task joint at its planned value per row, and of the arm to its own chassis. While
    the gripper is closed, the hand and fingers are not checked against the task object (they hold it)."""

    def __init__(self, task, scene, robot):
        import smoke_system as smoke
        cols = smoke.collisions(robots.current().model)
        self.shapes = {}
        for ln, cs in cols.items():
            pts = [smoke.sample_points(kind, T, dims) for kind, T, dims in cs]
            self.shapes[ln] = (np.vstack([p for p, _ in pts]), np.concatenate([r for _, r in pts]))
        self.chassis = next((T, np.array(d)) for kind, T, d in cols["base_link"] if kind == "box")
        self.robot, self.task = robot, task
        self.static, self.obj = [], None
        for name, uri, Tm, inline in sdf.world_entities(spec.path(scene["world"])):
            if name in scene["skip"]:
                continue
            if name == task["object"]:
                m = sdf.Model(spec.model_dir(scene["assets"], uri) / "model.sdf")
                self.obj = (Tm, m)
            elif inline is not None or name in scene["detailed"]:
                m = inline or sdf.Model(spec.model_dir(scene["assets"], uri) / "model.sdf")
                self.static += [(f"{name}::{c}", Tm @ m.pose(ln, {}) @ Tc, np.array(s))
                                for ln, bs in m.boxes.items() for c, Tc, s in bs]
            else:
                lo, hi = np.array(json.loads((spec.model_dir(scene["assets"], uri) / "manifest.json")
                                             .read_text())["aabb_model_frame"])
                self.static.append((f"{name}::aabb", Tm @ sdf.tf((lo + hi) / 2), hi - lo))

    def obj_boxes(self, door):
        Tm, m = self.obj
        return [(f"{self.task['object']}::{c}", Tm @ m.pose(ln, {self.task['joint']: door}) @ Tc, np.array(s))
                for ln, bs in m.boxes.items() for c, Tc, s in bs]

    @staticmethod
    def sd_box(p, T, size):
        q = np.abs((p - T[:3, 3]) @ T[:3, :3]) - size / 2
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)

    def at(self, b, q, g, door, holding):
        """{link: (min distance, obstacle)} for one robot state."""
        rb = robots.current()
        qd = dict(zip(rb.base_joints, b), **rb.gripper.joint_targets(g), **dict(zip(rb.arm_joints, q)))
        Tc = self.robot.pose("base_link", qd) @ self.chassis[0]
        objb = self.obj_boxes(door)
        out = {}
        for ln, (pts, rad) in self.shapes.items():
            Tl = self.robot.pose(ln, qd)
            p = pts @ Tl[:3, :3].T + Tl[:3, 3]
            cands = self.static + ([] if holding and ln in rb.gripper.hand_links else objb) + \
                ([] if ln in rb.on_chassis else [(f"{rb.model}::chassis", Tc, self.chassis[1])])
            best = (math.inf, "")
            for name, Tb, size in cands:
                d = float((self.sd_box(p, Tb, size) - rad).min())
                if d < best[0]:
                    best = (d, name)
            out[ln] = best
        return out


def clearance(P, door_rows, holding_rows, C):
    """Per event label: min clearance [m], link, obstacle, t; plus the base_link minimum. door_rows: the
    task joint per plan row (closed before the pull, the planned value during it, final after)."""
    starts = [(round(t / PLAN_DT), lab) for t, lab, _ in P.events]
    res = {}
    for i, (k0, lab) in enumerate(starts):
        k1 = starts[i + 1][0] if i + 1 < len(starts) else len(P.rows)
        for k in list(range(k0, k1, CLEAR_EVERY)) + [k1 - 1]:
            if k < k0 or k >= len(P.rows):
                continue
            b, q, g, _ = P.rows[k]
            holding = bool(holding_rows[k])
            for ln, (d, obs) in C.at(b, q, g, door_rows[k], holding).items():
                key = "base_link" if ln == "base_link" else "arm"
                for kk in (lab, f"{key}_all"):
                    if kk not in res or d < res[kk][0]:
                        res[kk] = [d, ln, obs, round(k * PLAN_DT, 2), lab]
    return {lab: dict(min_m=round(v[0], 4), link=v[1], obstacle=v[2], t=v[3], event=v[4]) for lab, v in res.items()}


def actions(plan, attempt, out):
    """actions.npz + episode.json of the regenerated part and its clearance precheck. Exit 0 = passed."""
    plan, out, attempt = Path(plan), Path(out), int(attempt)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((plan / "handoff.json").read_text())
    r = spec.task(meta["plugins"]["task"])
    task, scene = spec.legacy_task(r), spec.legacy_scene(r)
    gen = regen_settings()
    robot = sdf.Model(os.environ.get("SIM_ROBOT_MODEL") or spec.path(r.robot.camera_model))
    robot.zero = functools.lru_cache(maxsize=None)(robot.zero)
    B, facts, door_plan, rg, snap = build(plan, attempt, r, task)
    P = B.P
    seq = P.actions()
    seq.save(out / "actions.npz")
    sub, cur = [], rp_.SUBTASKS[0]
    for _, lab, _ in P.events:
        cur = {**rp_.SUBTASK_START, "reach": "approach", "check grasped": "grasp"}.get(lab, cur)
        sub.append(cur)
    obj, part, unit = spec.hold(r)[:3]
    rp = task["replay"]
    man = json.loads((spec.path(task["gz_model"]) / "manifest.json").read_text())
    hd = next(x for x in man["handles"] if x["name"] == task["handle"])
    ep = dict(facts, control_dt=CONTROL_DT, steps=len(seq), duration_s=P.t, budget_s=rp["budget_s"],
              contact_rate_hz=spec.SIM["contact_rate_hz"], task=task["name"], object_joint=f"{task['object']}::{task['joint']}",
              object_handle=f"{task['object']}::{hd['link']}::{task['handle']}", joint_topic=task["joint_topic"],
              prompt=task["prompt"], hold=dict(check=rp["hold_check"], obj=obj, part=part, unit=unit),
              thresholds=dict(spec.legacy_thresholds(r), grasp_half_width_m=spec.grasp_half_width(r, hd["size"])),
              approach=dict(standoff_m=rp["standoff_m"], settle_s=rp["settle_s"], approach_s=rp["approach_s"],
                            retreat_m=rp["retreat_m"]),
              events=[[round(t, 6), lab, info] for t, lab, info in P.events], event_subtask=sub,
              subtasks=list(rp_.SUBTASKS), checks=[[i, kind, info] for i, kind, info in B.checks],
              regenerated=dict(source_rollout=rg["source_rollout"], source_plan_run=rg.get("source_plan_run"),
                               source_attempt_id=rg.get("source_attempt_id"), phase=rg["phase"],
                               restore_phase=rg["restore_phase"], mode=rg["mode"],
                               action_index_offset=rg["action_index"], source_sim_time_s=rg["action_index"] * CONTROL_DT,
                               planner_seed=meta["seed"], planner_run=rg["planner_run"],
                               **({"grasp_from_state": True} if rg.get("grasp_from_state") else {})))
    # the task joint per plan row: as restored before the pull, the planned path during it, final after
    pull = next(i for i, (_, lab, _) in enumerate(P.events) if lab == "pull")
    k_pull = round(P.events[pull][0] / PLAN_DT)
    n_pull = round(ep["timing"]["pull_s"] / PLAN_DT)
    d0 = snap["joints"][ep["object_joint"]]
    t_wp = np.linspace(0, n_pull, len(door_plan) - 4)
    door_rows = np.full(len(P.rows), float(door_plan[-1]))
    door_rows[:k_pull] = d0
    door_rows[k_pull:k_pull + n_pull] = np.interp(np.arange(n_pull), t_wp, door_plan[4:])
    rel = next(i for i, (_, lab, _) in enumerate(P.events) if lab == "release")
    k_rel_end = round(P.events[rel + 1][0] / PLAN_DT)
    k_close = next((round(t / PLAN_DT) for t, lab, _ in P.events if lab == "grip close"), 0)
    holding = np.zeros(len(P.rows), bool)
    holding[k_close:k_rel_end] = True          # the hand on the handle: grip close (or the restart) to the release end
    C = Clearance(task, scene, robot)
    ep["clearance"] = clearance(P, door_rows, holding, C)
    ep["clearance"]["settle"] = dict(ep["clearance"][P.events[0][1]])     # the restored start: spawn check
    # base margin: clearance_min_m, but never more than the restored start itself keeps (minus
    # start_slack_m): the source may have left the chassis closer than that (drawer r2_a37: 17.5 mm)
    b0, q0, g0, _ = P.rows[0]
    start_base = C.at(b0, q0, g0, door_rows[0], bool(holding[0]))["base_link"][0]
    gate = dict(gen["clearance_min_m"])
    gate["base_link"] = round(max(0.0, min(gate["base_link"], start_base - gen["start_slack_m"])), 4)
    ep["clearance"]["start_base_link_m"] = round(start_base, 4)
    worst = {k: ep["clearance"][f"{k}_all"] for k in ("base_link", "arm")}
    ok = all(worst[k]["min_m"] >= gate[k] for k in worst)
    ep["clearance"]["precheck_gates_m"] = gate
    (out / "episode.json").write_text(json.dumps(ep, indent=1, default=float))
    print(f"regen attempt {attempt}: {len(seq)} steps ({P.t:.1f} s), pull {ep['timing']['pull_s']:.1f} s, planned "
          f"{part} {facts[f'planned_{part}_final_{unit}']:.1f} {unit}, start jump {facts['start_jump_max']:.4f}; "
          + ", ".join(f"{k} min {v['min_m'] * 1000:.1f} mm ({v['link']} / {v['obstacle']} at {v['t']} s, {v['event']})"
                      for k, v in worst.items())
          + f" -> precheck {'PASS' if ok else 'FAIL'} (gates {gate})")
    (out / "precheck.json").write_text(json.dumps(dict(pass_=ok, gates=gate, worst=worst), indent=1))
    return 0 if ok else 1


def prepare(plan, out):
    """live.json (replay mode, restored in step 0) for the candidate in out; phases from restore_phase on."""
    plan, out = Path(plan), Path(out)
    ep = json.loads((out / "episode.json").read_text())
    meta = json.loads((plan / "handoff.json").read_text())
    rg = meta["regenerate"]
    h = np.load(plan / "handoff.npz")
    k = int(np.flatnonzero(h["attempt_id"] == ep["attempt_id"])[0])
    r = spec.task(ep["task"])
    phase = rg["restore_phase"]
    (out / "restore").mkdir(exist_ok=True)
    shutil.copy(plan / "start_state.npz", out / "restore" / f"{phase}_start.npz")
    shutil.copy(plan / "start_state_next.npz", out / "restore" / f"{phase}_start_next.npz")
    snap = np.load(plan / "start_state.npz")
    live = live_validate.live_json(r, out, f"{rg['source_label']} recovered from {rg['phase']} ({plan.name}/{ep['attempt_id']})",
                                   "1khz", h["tcp_world"][k, 0], "replay", plan_run=str(plan), attempt_id=ep["attempt_id"],
                                   regenerated=True)
    live["restore"] = dict(phase=phase, action_index=rg["action_index"], snapshot=f"restore/{phase}_start.npz",
                           snapshot_next=f"restore/{phase}_start_next.npz", source_run=rg["source_rollout"],
                           **({"grasp_rel": snap["grasp_rel"].tolist()} if "grasp_rel" in snap.files else {}))
    (out / "live.json").write_text(json.dumps(live, indent=1))
    print(f"prepared {live['label']}: {ep['steps']} steps -> {out}")


def candidate(a):
    out, plan = cli.rpath(a.out), cli.rpath(a.plan)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    r = spec.task(json.loads((plan / "handoff.json").read_text())["plugins"]["task"])
    cam, _ = cli.models(r)
    os.environ["SIM_ROBOT_MODEL"] = str(cam / "model.sdf")
    os.environ["GZ_SIM_RESOURCE_PATH"] = f"{cam.parent}:{spec.resource_path(r)}"   # the precheck reads models
    cli.progress(stage="actions", plan=plan.name, attempt=a.attempt)
    if actions(plan, a.attempt, out) != 0:
        print("clearance precheck failed: rejected before gz")
        return 2
    prepare(plan, out)
    rc = cli.phase_run(out, r, "RECOVER", cameras=a.cameras, timeout=600)
    if rc in (0, 1) and a.cameras and not a.no_bag:
        cli.progress(stage="bag")
        subprocess.run([sys.executable, str(cli.SIM / "bag.py"), "write", str(out)], check=True)
        chk = subprocess.run([sys.executable, str(cli.SIM / "bag.py"), "check", str(out / "bag"), str(out)],
                             capture_output=True, text=True)
        (out / "bag_check.json").write_text(chk.stdout)
        if chk.returncode:
            print(f"bag check FAILED: {chk.stdout[-500:]}")
            rc = 4
    return rc


# ------------------------------------------------------------------ test support: a policy-like open start
def grasp_rel_of(snap, obj):
    """The handle link in the TCP frame (phase_replay_system.handle_rel) from a snapshot's link poses."""
    import scene_state as ss
    from skill_system import OBJ
    mn, ln, _ = OBJ[obj]
    names = [str(x) for x in snap["link_names"]]

    def pose(name):
        i = names.index(name)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = ss.quat_mat(snap["link_quat"][i]), snap["link_pos"][i]
        return T
    rb = robots.current()
    tcp = pose(f"{rb.model}::{rb.hand_link}") @ tf(p=(0, 0, rb.tcp_z))
    return np.linalg.inv(tcp) @ pose(f"{mn}::{ln}")


def perturb(a):
    src, out = cli.rpath(a.source), cli.rpath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    run = out / "gz"
    live_validate.prepare_restore(src, "open", run)
    live = json.loads((run / "live.json").read_text())
    r = spec.task(live["task"])
    si = spec.hold(r)[3]
    b = live["restore"]["action_index"]
    q = np.load(src / "state_log.npz")["door_pos"][b:]
    m = int(np.argmax(q >= si(a.joint)))
    if q[m] < si(a.joint):
        raise SystemExit(f"{src} never reaches {a.joint} after its open start (max {q.max()})")
    A = dict(np.load(run / "actions.npz"))
    k = m + int(round(a.settle_s / CONTROL_DT))
    for f in ("base", "arm", "grip", "grip_force"):
        A[f][m:] = A[f][m]
    for f in ("base_vel", "arm_vel"):
        A[f][m:] = 0.0
    A["grip"][k:] = robots.current().gripper.open_m
    np.savez_compressed(run / "actions.npz", **A)
    print(f"perturb: pull fed to step {m} ({a.joint} {live['unit']}), held {a.settle_s} s, gripper open at {k}")
    cli.models(r)
    rc = cli.phase_run(run, r, "PERTURB", timeout=600)
    rs = dict(np.load(run / "phase_states" / "release_start.npz", allow_pickle=True))
    if int(rs["step"]) != k:
        raise SystemExit(f"release boundary at step {rs['step']}, not {k} (driver rc {rc})")
    ro = out / "rollout"
    (ro / "phase_states").mkdir(parents=True)
    for name in ("release_start", "release_start_next"):
        s = dict(np.load(run / "phase_states" / f"{name}.npz", allow_pickle=True))
        s.update(action_grip=A["grip"][k - 1], action_grip_force=A["grip_force"][k - 1], phase=np.asarray("open"),
                 grasp_rel=grasp_rel_of(rs, live["obj"]))
        np.savez_compressed(ro / "phase_states" / f"{name.replace('release', 'open')}.npz", **s)
    q_si = float(dict(zip([str(x) for x in rs["joint_names"]], rs["joint_pos"]))[f"{r.task.object.name}::{r.task.object.joint}"])
    ep = json.loads((src / "episode.json").read_text())
    pr = dict(result=dict(failed_phase="open", regenerate=True, failed_checks=["perturbed"], first_failure_t=None),
              phases={}, note=f"test fixture: {src.name} open start, pull stopped at {a.joint} {live['unit']}")
    (ro / "episode.json").write_text(json.dumps(dict(task=r.task.name, prompt=ep.get("prompt"), control_dt=CONTROL_DT,
                                                     seed=(ep.get("datagen") or {}).get("seed"),
                                                     policy_rollout=dict(perturbed_from=str(src), stop_at=a.joint,
                                                                         joint_si=q_si)), indent=1))
    (ro / "phase_report.json").write_text(json.dumps(pr, indent=1))
    print(f"perturbed open start: {r.task.object.joint} {q_si:.4f} (SI) at source action {int(rs['action_index'])} -> {ro}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidate")
    c.add_argument("--plan", required=True)
    c.add_argument("--attempt", required=True, type=int)
    c.add_argument("--out", required=True)
    c.add_argument("--cameras", type=int, default=0, help="training camera rate Hz (0 = render off)")
    c.add_argument("--no-bag", action="store_true")
    p = sub.add_parser("perturb")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--joint", required=True, type=float, help="stop the pull here (the task's unit: deg or cm)")
    p.add_argument("--settle-s", type=float, default=0.5)
    a = ap.parse_args(argv)
    return {"candidate": candidate, "perturb": perturb}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
