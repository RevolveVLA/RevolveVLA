"""Plan the rest of a task from a saved phase start state (recovery entry.py plan, in revolve-recovery).
Ported from the earlier pipeline's plan_from_state.py: the specs via datagen's specs_bridge (was
its plugin loader), the task named in regen.json (was the source plan run's handoff.json), the
datagen component's planner modules (plan.py, gen_assets.py, frames.py). The algorithm is unchanged.

  python plan_from_state.py <plan dir>      env SEED, ATTEMPTS (default: the generator's regenerate: block)

<plan dir> holds (recover.py writes them): regen.json (task, phase, mode, the source rollout and its
attempt and grasp, the action index of the restart) and start_state.npz (the rollout's
phase_states/<phase>_start.npz: scene_state.capture of the whole scene + the commanded action).
The planner setup is plan.py's (gen_assets, setup_env, the warp mesh cache fix, the trajopt pose
weights), with the regeneration's extra chassis corner spheres (generator `regenerate:`).

  mode pull   (an open-phase restart: the robot holds the handle) one start = the saved robot
              joints with the saved door / drawer position on the source attempt's grasp AKR;
              goals = plan_ik at the goal joint value for that grasp, the ATTEMPTS raw solutions
              nearest to the start (regenerate.goal_spacing apart); plan_traj + filter_traj as plan.py.
              regen.json grasp_from_state (a policy rollout): the grasp is the one the saved state
              holds (its grasp_rel), added as the last grasp id, instead of the source attempt's.
  mode reach  (an approach or grasp restart: not grasping) plan.py's IK, clustering, pairs, AKR
              trajopt and filter for ATTEMPTS attempts over the grasps; then per planned pull a
              cuRobo MotionGen reach (whole-body cspace, plan_single_js) from the saved robot
              joints to the pull start moved approach.standoff_m back along the hand axis (the
              staging pose replay.py spawns at).

Writes handoff.npz / handoff.json in plan.py's format (one row per attempt, rows 0-3 the grasp,
4-35 the pull) plus reach_world (A, R, 3 + arm DoF) / reach_len / reach_dt for mode reach, handoff.json
"regenerate" (regen.json + the corner spheres), plan_report.json and assets/. The robot is the task spec's
($REVOLVE_ROBOT); its chassis corner spheres are robot.yaml planner regen_base_corner_spheres.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import frames
import gen_assets
import specs_bridge as plugins
from plan import convert_12d, now, pick_pairs

def world_to_obj_base(b, obj, yaw_lo):
    """World base (x, y, yaw) -> the planner (object) frame, yaw in [yaw_lo, yaw_lo + 2 pi)."""
    T = frames.object_tf(obj)
    xy = T[:2, :2].T @ (np.asarray(b[:2]) - T[:2, 3])
    yaw = (b[2] - obj[3] - yaw_lo) % (2 * np.pi) + yaw_lo
    return np.array([xy[0], xy[1], yaw])


def main(out):
    out = Path(out).resolve()
    rg = json.loads((out / "regen.json").read_text())
    r = plugins.task_spec(rg["task"])
    names = dict(robot=r.robot.name, scene=r.scene.name, task=r.task.name, generator="automoma")
    rd, sc, task, gen = (plugins.load(k, names[k]) for k in ("robot", "scene", "task", "generator"))
    task = dict(task, replay=r.task.replay.model_dump())      # the reach's standoff_m
    R = gen["regenerate"]
    rd = dict(rd, base_corner_spheres=rd["regen_base_corner_spheres"], base_self_collision_buffer=R["base_self_collision_buffer"])
    rb = gen_assets.gz_robot(rd)
    nq = 3 + rb.dof                              # base + arm joints
    attempts = int(os.environ.get("ATTEMPTS", R["attempts"]))
    seed = int(os.environ.get("SEED", R["seed0"]))
    torch.manual_seed(seed)
    np.random.seed(seed)
    # grasp_from_state (a policy rollout's open restart): the pull's AKR is built for the grasp the saved
    # state actually holds (grasp_rel = the handle link in the TCP frame), appended to the task's grasps
    s = np.load(out / "start_state.npz", allow_pickle=True)
    if rg.get("grasp_from_state"):
        task = dict(task, grasp=dict(task["grasp"], measured_tcp_in_link=[np.linalg.inv(s["grasp_rel"]).tolist()]))
    from curobo.wrap.reacher.motion_gen import MotionGenConfig, MotionGenPlanConfig
    load = MotionGenConfig.load_from_robot_config
    seeds = dict(ik_seed=1531 + seed, graph_seed=1531 + seed)
    MotionGenConfig.load_from_robot_config = staticmethod(lambda *a, **k: load(*a, **seeds, **k))
    smi_log = open("/tmp/smi.csv", "w+")
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
                           stdout=smi_log)
    T = {}
    t0 = now()
    A = gen_assets.build(rd, sc, task, out / "assets")
    traj_file = None
    if gen.get("trajopt_pose_weight"):
        import yaml
        from curobo.util_file import get_task_configs_path, join_path, load_yaml
        tcfg = load_yaml(join_path(get_task_configs_path(), "gradient_trajopt_fixbase.yml"))
        tcfg["cost"]["pose_cfg"]["weight"] = gen["trajopt_pose_weight"]
        traj_file = out / "assets" / "gradient_trajopt_fixbase.yml"
        traj_file.write_text(yaml.safe_dump(tcfg, sort_keys=False))
    T["assets_s"] = now() - t0

    from automoma.core.types import TrajResult
    from automoma.planning.pipeline import PlanningPipeline
    from automoma.utils.file_utils import load_object_from_metadata, load_robot_cfg, process_robot_cfg
    from automoma.utils.math_utils import get_open_ee_pose, stack_iks_angle
    from curobo.types.math import Pose
    from curobo.types.state import JointState

    t = now()
    pipe = PlanningPipeline({"planner": gen["planner"]})
    planner = pipe.planner
    scene_cfg = {"path": str(A["scene"]["usd"]), "pose": [0, 0, 0, 1, 0, 0, 0],
                 "metadata_path": str(A["scene"]["metadata"])}
    obj_cfg = load_object_from_metadata(scene_cfg["metadata_path"], {
        "path": str(A["object"]["urdf"]), "asset_type": task["object"], "asset_id": task["object"]})
    planner.setup_env(scene_cfg, obj_cfg)
    T["setup_env_s"] = now() - t
    update = planner._update_world_collision

    def fresh_object_mesh(mg, *a, **k):         # plan.py: warp caches the object mesh by name
        mg.world_coll_checker._wp_mesh_cache.pop("/World/object/target_object", None)
        return update(mg, *a, **k)
    planner._update_world_collision = fresh_object_mesh
    robot_cfg = process_robot_cfg(load_robot_cfg(str(A["robot"]["yml"])))
    obj_pose = Pose.from_list(planner.object_pose)
    jn, goal = task["joint"], float(task["goal_angle"])
    unit = {"revolute": "rad", "prismatic": "m"}[A["object"]["joint_type"]]
    ax = A["object"]["grasp_axis"]
    closed = {jn: 0.0}
    obj = task["world_pose"]

    # the saved state: robot joints (measured) and the task joint, in the planner frame
    jp = {str(n): float(v) for n, v in zip(s["joint_names"], s["joint_pos"])}
    bw = np.array([jp[f"{rb.model}::{j}"] for j in rb.base_joints])
    arm = np.array([jp[f"{rb.model}::{j}"] for j in rb.arm_joints])
    door0 = jp[f"{task['object']}::{jn}"]
    yaw_lo = task["base_yaw_window"][0]
    q_start = np.r_[world_to_obj_base(bw, obj, yaw_lo), arm]
    print(f"start state: world base {bw.round(4).tolist()}, planner {q_start[:3].round(4).tolist()}, {jn} {door0:.6f}")

    def ik_at(gid, angle):
        g = A["object"]["grasps"][gid]
        target = get_open_ee_pose(obj_pose, Pose.from_list(g["pose_obj"]), planner.object_urdf, task["handle"],
                                  {jn: angle}, closed)
        r = planner.plan_ik(torch.tensor(target.to_list()), robot_cfg,
                            plan_cfg={"joint_cfg": {jn: angle}, "enable_collision": True})
        out_ = (len(r), r.iks[planner.cluster_ik(r)].cpu() if len(r) else r.iks.cpu(), r.iks.cpu())
        planner.free_cuda_cache()
        return out_

    t = now()
    problems = []          # (gid, S row (nq), G row (nq))
    ik = {}
    if rg["mode"] == "pull":
        gid = max(A["object"]["grasps"]) if rg.get("grasp_from_state") else rg["source_grasp_id"]
        # One fixed start: AutoMoMa's cluster representatives lie >= 0.95 rad (joint-space L2) from it
        # and trajopt failed on all 40 (fridge r1_a13), so the goals are the raw IK solutions nearest
        # to the start, each >= goal_spacing from the ones already taken.
        raw, Gc, G = ik_at(gid, goal)
        S0 = torch.tensor(q_start, dtype=G.dtype)[None]
        if len(G):
            dist = torch.cdist(S0, G)[0]
            pick = []
            for j in dist.argsort().tolist():
                if all(float(torch.linalg.norm(G[j] - G[i])) >= R["goal_spacing"] for i in pick):
                    pick.append(j)
                if len(pick) == attempts:
                    break
            pick = [pick[i % len(pick)] for i in range(attempts)]
            problems = [(gid, S0[0], G[j]) for j in pick]
            ik[gid] = dict(goal_raw=raw, goal_clustered=len(Gc), goal_dist_to_start=[round(float(dist[j]), 3) for j in pick])
        print(f"grasp {gid} ({ax} {A['object']['grasps'][gid][ax]:.4f}{', measured in the saved state' if rg.get('grasp_from_state') else ''}): "
              f"goal IK {raw}; {len(problems)} problems from the saved state, goals "
              f"{ik.get(gid, {}).get('goal_dist_to_start')} from it")
    else:
        for gid in A["object"]["grasps"]:
            (rs, Sg, _), (rg_, Gg, _) = ik_at(gid, 0.0), ik_at(gid, goal)
            ik[gid] = dict(start=Sg, goal=Gg, raw=(rs, rg_))
            print(f"grasp {gid}: IK start {rs} -> {len(Sg)}, goal {rg_} -> {len(Gg)}")
        ok = [g for g in ik if len(ik[g]["start"]) and len(ik[g]["goal"])]
        quota = {g: attempts // len(ok) + (i < attempts % len(ok)) for i, g in enumerate(ok)} if ok else {}
        for gid in ok:
            for si_, gi in pick_pairs(ik[gid]["start"], ik[gid]["goal"], quota[gid]):
                problems.append((gid, ik[gid]["start"][si_], ik[gid]["goal"][gi]))
    T["ik_s"] = now() - t
    if not problems:
        raise SystemExit("no problems: no IK")

    # An open restart starts where the source left the robot. When that state collides under the
    # corner spheres (drawer r2_a37: the chassis within 2.5 cm of the cabinet), trajopt cannot leave
    # it: plan the pull with the plain base spheres and leave the margin to the gz precheck.
    corners = dict(used=True)
    if rg["mode"] == "pull":
        valid, status = planner.motion_gen.check_start_state(
            JointState.from_position(torch.tensor(q_start, dtype=torch.float32, device="cuda")[None]))
        if not valid:
            A = gen_assets.build({k: v for k, v in rd.items() if k not in ("base_corner_spheres",
                                  "base_self_collision_buffer")}, sc, task, out / "assets_plain_base")
            corners = dict(used=False, reason=f"saved start invalid with the corner spheres: {status}")
            print(f"corner spheres off for the pull: {corners['reason']}")

    t = now()
    if traj_file:       # the AKR MotionGen only: the reach MotionGen (made by plan_ik above) keeps cuRobo's default
        planner.cfg["traj"]["gradient_trajopt_file"] = str(traj_file)
    rows, results = [], []
    for gid in dict.fromkeys(g for g, _, _ in problems):
        idx = [i for i, (g, _, _) in enumerate(problems) if g == gid]
        d0 = -door0 if rg["mode"] == "pull" else 0.0
        S = stack_iks_angle(torch.stack([problems[i][1] for i in idx]), d0)
        G = stack_iks_angle(torch.stack([problems[i][2] for i in idx]), -goal)
        akr = process_robot_cfg(load_robot_cfg(A["object"]["grasps"][gid]["akr_yml"]))
        tr = planner.plan_traj(S, G, akr, plan_cfg={"expand_to_pairs": False, "batch_size": min(len(idx), gen["planner"]["traj"]["batch_size"]),
                                                    "enable_collision": True})
        for j, i in enumerate(idx):
            one = tr[j:j + 1]
            passed = bool(one.success.any()) and planner.filter_traj(one, akr).num_samples > 0
            rows.append(dict(attempt_id=i, grasp_id=gid, **{f"grasp_{ax}": A["object"]["grasps"][gid][ax]},
                             trajopt_success=bool(one.success.any()), filter_pass=passed, success=passed))
            results.append((i, TrajResult(one.start_states, one.goal_states, one.trajectories, torch.tensor([passed]))))
        planner.free_cuda_cache()
    planner.cfg["traj"].pop("gradient_trajopt_file", None)
    results.sort(key=lambda x: x[0])
    rows.sort(key=lambda r: r["attempt_id"])
    T["traj_s"] = now() - t

    conv = convert_12d(pipe, TrajResult.cat([r for _, r in results]), nq)
    (out / "automoma").mkdir(parents=True, exist_ok=True)
    pipe.planning_io.save_converted(conv, str(out / "automoma" / "traj_data_train.pt"))
    traj = conv["traj_robot"].numpy().astype(np.float64)
    q = torch.as_tensor(traj[..., :nq].reshape(-1, nq), dtype=torch.float32, device="cuda")
    ee = planner.motion_gen.ik_solver.fk(q).ee_pose
    tcp_obj = frames.pose7_to_tf(torch.cat([ee.position, ee.quaternion], -1).cpu().numpy().astype(np.float64))
    tcp_world = frames.pose_to_world(tcp_obj, obj).reshape(*traj.shape[:2], 4, 4)
    world = traj.copy()
    world[..., :3] = frames.base_to_world(traj[..., :3], obj)
    door = conv["traj_obj"].numpy()[..., 0].astype(np.float64)
    vel = planner.motion_gen.kinematics.kinematics_config.joint_limits.velocity[1].cpu().numpy()
    for r in rows:
        w = world[r["attempt_id"]]
        d = np.diff(w[:, :nq], axis=0)
        seg = np.maximum(np.linalg.norm(d[:, :2], axis=1) / vel[0], np.abs(d[:, 2:nq] / vel[2:nq]).max(1))
        r.update(duration_hint_s=float(seg.sum()), base_travel_m=float(np.linalg.norm(d[:, :2], axis=1).sum()),
                 base_start_world=w[0, :3].tolist(), base_end_world=w[-1, :3].tolist(),
                 start_vs_saved_max=float(np.abs(traj[r["attempt_id"], 0, :nq] - q_start).max()),
                 **{f"door_final_{unit}": float(door[r["attempt_id"], -1])})

    # reach (mode reach): MotionGen from the saved state to each planned pull's staging pose
    reach = {}
    if rg["mode"] == "reach":
        t = now()
        rp = task["replay"]
        mg = planner.motion_gen                     # the IK MotionGen: whole-body cspace, scene + closed object
        planner._update_world_collision(mg, {jn: door0}, True, include_object_mesh=True)
        start = JointState.from_position(torch.tensor(q_start, dtype=torch.float32, device="cuda")[None])
        for r in rows:
            if not r["success"]:
                continue
            q0 = torch.as_tensor(traj[r["attempt_id"], 0, :nq], dtype=torch.float32, device="cuda")[None]
            Rz = frames.quat_wxyz_to_R(planner.motion_gen.ik_solver.fk(q0).ee_pose.quaternion.cpu().numpy()[0])[:, 2]
            axy = Rz[:2] / np.linalg.norm(Rz[:2])
            stage = q0.clone()
            stage[0, :2] -= torch.tensor(rp["standoff_m"] * axy, dtype=torch.float32, device="cuda")
            res = mg.plan_single_js(start, JointState.from_position(stage),
                                    MotionGenPlanConfig(max_attempts=R["reach_max_attempts"], enable_graph=True))
            ok = bool(res.success.item()) if res.success is not None else False
            r.update(reach_success=ok, reach_status=str(res.status))
            if ok:
                p = res.get_interpolated_plan().position.cpu().numpy().astype(np.float64)
                pw = p.copy()
                pw[:, :3] = frames.base_to_world(p[:, :3], obj)
                reach[r["attempt_id"]] = (pw, float(res.interpolation_dt))
                r.update(reach_steps=len(p), reach_dt_s=float(res.interpolation_dt),
                         reach_s=float(res.interpolation_dt) * (len(p) - 1),
                         reach_end_err=float(np.abs(p[-1] - stage.cpu().numpy()[0]).max()))
            else:
                r["success"] = False
            print(f"attempt {r['attempt_id']}: reach {'ok' if ok else 'FAILED'} {r['reach_status']}"
                  + (f" {r['reach_s']:.2f} s" if ok else ""))
        T["reach_s"] = now() - t

    smi.terminate()
    smi.wait()
    smi_log.seek(0)
    smi_mib = [int(x) for x in smi_log.read().split() if x.isdigit()]
    success = np.array([r["success"] for r in rows])
    arrays = dict(joint_names=np.array(robot_cfg["kinematics"]["cspace"]["joint_names"]), traj_robot_obj=traj,
                  traj_robot_world=world, door_angle=door, tcp_world=tcp_world, success=success,
                  attempt_id=np.arange(len(rows)), grasp_id=np.array([r["grasp_id"] for r in rows]))
    if rg["mode"] == "reach":
        n = max([len(v[0]) for v in reach.values()], default=1)
        rw, rl, rdt = np.zeros((len(rows), n, nq)), np.zeros(len(rows), int), np.zeros(len(rows))
        for a, (p, dt) in reach.items():
            rw[a, :len(p)], rl[a], rdt[a] = p, len(p), dt
        arrays.update(reach_world=rw, reach_len=rl, reach_dt=rdt)
    np.savez_compressed(out / "handoff.npz", **arrays)
    meta = {"format": f"plan.py handoff format (handoff.json of a plan run); reach_world (A, R, {nq}) world base + arm "
                      "at reach_dt (A,) s, reach_len (A,) valid rows (mode reach only)",
            "run": out.name, "plugins": names, "seed": seed, "curobo_seeds": seeds, "attempts": len(rows),
            "planned": int(success.sum()), "object_world_pose": obj, "goal_angle": goal,
            "grasps": {g: {k: v[k] for k in (ax, "pose_obj", "T_handle_grasp")} for g, v in A["object"]["grasps"].items()},
            "attempts_detail": rows,
            "regenerate": dict(rg, base_corner_spheres=rd["base_corner_spheres"],
                               base_self_collision_buffer=R["base_self_collision_buffer"], corner_spheres=corners, reach_slowdown=R["reach_slowdown"],
                               start_state_world_base=bw.tolist(), start_state_planner=q_start.tolist(),
                               start_task_joint=door0)}
    (out / "handoff.json").write_text(json.dumps(meta, indent=1, default=float))
    T["total_s"] = now() - t0
    report = dict(mode=rg["mode"], attempts=len(rows), planned=int(success.sum()),
                  trajopt_success=int(sum(r["trajopt_success"] for r in rows)),
                  reach_success=int(sum(r.get("reach_success", False) for r in rows)), ik=ik and {
                      str(g): {k: (v if not torch.is_tensor(v) else len(v)) for k, v in d.items()} for g, d in ik.items()},
                  times=T, seed=seed, curobo_seeds=seeds, vram_peak_mib=max(smi_mib) if smi_mib else None,
                  robot_base_spheres=len(A["robot"]["kinematics"]["collision_spheres"]["base_link"]),
                  corner_spheres=len(A["robot"]["kinematics"]["collision_spheres"].get("base_corners", [])),
                  corner_spheres_in_pull=corners)
    (out / "plan_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
