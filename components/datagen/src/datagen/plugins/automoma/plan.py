"""The plan run, in the datagen image (cuRobo + AutoMoMa, GPU):

  python -m datagen.plugins.automoma.plan --instance <scene.yaml> --task T --robot R --out <scene dir>

Plans the whole-body motion of one scene instance with AutoMoMa's own classes (CuroboPlanner: IK, AKR trajopt, FK filter): a grasp
IK nearest the instance's start base, the goal IK at the open door (seeded with the start moved rigidly with the handle), the
pull as a trajopt of the robot + door chain, per grasp point along the handle until one passes; then cuRobo MotionGen's drive-in
from the start (arm folded) to the staging pose behind the grasp. Writes <out>/plan.json (`ok`, `reason` or the 30 Hz series of
series.py) and <out>/plan/ (assets, report). Exit 0: planned, 3: no plan (the reason is in plan.json)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from . import assets, frames, problem, series

EXPERT = "automoma_curobo_v1"
IK_SEEDS = 1000             # IK seeds of a seeded IK (start base nearest)
DRIVE_IN_MAX_DT = 0.5       # s: the drive-in MotionGen's trajopt step cap (32 steps: up to 15.5 s)


class NoPlan(Exception):
    """The reason this instance has no plan."""


def seeded_ik(planner, target, seed):
    """The whole-body IK (base + arm) of a TCP target Pose nearest `seed` (a list), regularised toward it; None without one."""
    import torch
    s = planner.tensor_args.to_device(seed)
    r = planner.motion_gen.ik_solver.solve_single(target, retract_config=s[None], seed_config=s[None, None],
                                                  return_seeds=IK_SEEDS, num_seeds=IK_SEEDS)
    sol = r.solution.reshape(-1, s.shape[0])[r.success.reshape(-1)]
    return sol[int(torch.linalg.norm(sol - s, dim=1).argmin())].cpu() if len(sol) else None


def goal_seed(q, tcp_start, tcp_goal, yaw_window):
    """Goal IK seed: the base of q moved rigidly with the TCP from its start to its goal pose (planner frame, 4x4), the yaw
    kept inside the window, the arm of q."""
    import torch
    M = tcp_goal @ np.linalg.inv(tcp_start)
    x, y, yaw = (float(v) for v in q[:3])
    B = M @ frames.object_tf((x, y, 0.0, yaw))
    yaw_g = min(max(yaw + float(np.arctan2(M[1, 0], M[0, 0])), yaw_window[0] + 1e-3), yaw_window[1] - 1e-3)
    return torch.cat([torch.tensor([B[0, 3], B[1, 3], yaw_g], dtype=q.dtype), q[3:]])


def drive_motion_gen(planner, robot_cfg, jn):
    """MotionGen in the whole-body cspace with the scene and the closed object; DRIVE_IN_MAX_DT (a drive-in is long: cuRobo's
    default step cap ends it in DT_EXCEPTION)."""
    from curobo.wrap.reacher.motion_gen import MotionGenConfig
    traj_cfg = planner.cfg.setdefault("traj", {})
    grad = traj_cfg.pop("gradient_trajopt_file", None)
    load = MotionGenConfig.load_from_robot_config
    MotionGenConfig.load_from_robot_config = staticmethod(
        lambda *a, **k: load(*a, **{"maximum_trajectory_dt": DRIVE_IN_MAX_DT, **k}))
    try:
        mg = planner.init_motion_gen(robot_cfg, fixed_base=False, collision_role="ik", enable_collision=True)
    finally:
        MotionGenConfig.load_from_robot_config = load
        if grad is not None:
            traj_cfg["gradient_trajopt_file"] = grad
    planner._update_world_collision(mg, {jn: 0.0}, True, include_object_mesh=True)
    return mg


def drive_in(planner, robot_cfg, jn, start, q_grasp, standoff, obj):
    """-> (world rows (R, 10), dt) of the drive-in: whole-body MotionGen plan_single_js from `start` (planner frame base + arm) to
    the staging pose (q_grasp's base moved `standoff` back along the hand axis in the floor plane); NoPlan with cuRobo's status."""
    import torch
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    mg = drive_motion_gen(planner, robot_cfg, jn)
    dev = dict(dtype=torch.float32, device="cuda")
    q0 = torch.as_tensor(np.asarray(q_grasp, float), **dev)[None]
    Rz = rotation(mg.ik_solver.fk(q0).ee_pose.quaternion.cpu().numpy()[0])[:, 2]
    stage = q0.clone()
    stage[0, :2] -= torch.tensor(standoff * Rz[:2] / np.linalg.norm(Rz[:2]), **dev)
    s = torch.as_tensor(np.asarray(start, float), **dev)[None]
    res = mg.plan_single_js(JointState.from_position(s), JointState.from_position(stage),
                            MotionGenPlanConfig(max_attempts=4, enable_graph=True))
    if res.success is None or not bool(res.success.item()):
        raise NoPlan(f"drive-in from the start failed: {res.status}")
    p = res.get_interpolated_plan().position.cpu().numpy().astype(np.float64)
    p[:, :3] = frames.base_to_world(p[:, :3], obj)
    del mg
    planner.free_cuda_cache()
    return p, float(res.interpolation_dt)


def rotation(q):
    """3x3 of a unit quaternion w, x, y, z."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def plan_pull(planner, p, grasp, gid, tcp_closed, tcp_goal, q_start):
    """-> (rows (32, 10) planner frame, the grasp IK) of the pull at grasp `gid`: grasp IK nearest `q_start`, goal IK seeded with it,
    trajopt of the robot + door chain, AutoMoMa's FK filter. NoPlan with the stage that failed."""
    from automoma.utils.file_utils import load_robot_cfg, process_robot_cfg
    from automoma.utils.math_utils import stack_iks_angle
    jn, goal = p.joint, p.goal
    planner._update_world_collision(planner.motion_gen, {jn: 0.0}, True, include_object_mesh=True)
    q_g = seeded_ik(planner, tcp_closed, q_start)
    if q_g is None:
        raise NoPlan(f"grasp {gid}: no collision-free whole-body IK at the closed door")
    planner._update_world_collision(planner.motion_gen, {jn: goal}, True, include_object_mesh=True)
    yaw0 = p.start_base_obj[2]
    window = (yaw0 - p.yaw_window, yaw0 + p.yaw_window)
    mats = [t.get_matrix()[0].cpu().numpy().astype(np.float64) for t in (tcp_closed, tcp_goal)]
    q_e = seeded_ik(planner, tcp_goal, goal_seed(q_g, *mats, window).tolist())
    if q_e is None:
        raise NoPlan(f"grasp {gid}: no collision-free whole-body IK at the open door")
    akr = process_robot_cfg(load_robot_cfg(grasp["akr_yml"]))
    tr = planner.plan_traj(stack_iks_angle(q_g[None], 0.0), stack_iks_angle(q_e[None], -goal), akr,
                           plan_cfg={"expand_to_pairs": False, "enable_collision": True, "batch_size": 1})
    if not bool(tr.success.any()):
        raise NoPlan(f"grasp {gid}: trajopt failed")
    kept = planner.filter_traj(tr, akr)
    if kept.num_samples == 0:
        raise NoPlan(f"grasp {gid}: the planned pull drifts off the handle (FK filter)")
    rows = kept.trajectories[0, :, :10].numpy().astype(np.float64)
    return rows, q_g


def setup(p, work: Path):
    """The assets of the problem and AutoMoMa's planner on them -> (planner, the robot cfg dict, the object cfg of the assets)."""
    from automoma.planning.pipeline import PlanningPipeline
    from automoma.utils.file_utils import load_object_from_metadata, load_robot_cfg, process_robot_cfg
    from curobo.util_file import get_task_configs_path, join_path, load_yaml
    gen = copy.deepcopy(p.planner)
    chain = assets.franka_chain(p.cfg, p.tcp)
    rob = assets.robot(p, chain, work / "assets")
    obj = assets.task_object(p, p.object_model(), rob, work / "assets")
    scn = assets.scene(p, work / "assets")
    tcfg = load_yaml(join_path(get_task_configs_path(), "gradient_trajopt_fixbase.yml"))
    tcfg["cost"]["pose_cfg"]["weight"] = gen["trajopt_pose_weight"]
    tf = work / "assets" / "gradient_trajopt_fixbase.yml"
    tf.write_text(yaml.safe_dump(tcfg, sort_keys=False))
    gen["planner"]["traj"]["gradient_trajopt_file"] = str(tf)           # absolute: cuRobo uses it as is
    planner = PlanningPipeline({"planner": gen["planner"]}).planner
    meta = {"path": str(obj["urdf"]), "asset_type": p.object, "asset_id": p.object}
    planner.setup_env({"path": str(scn["usd"]), "pose": [0, 0, 0, 1, 0, 0, 0], "metadata_path": str(scn["metadata"])},
                      load_object_from_metadata(str(scn["metadata"]), meta))
    update = planner._update_world_collision                             # cuRobo's warp cache keys meshes by name: drop it first

    def fresh_object_mesh(mg, *a, **k):
        mg.world_coll_checker._wp_mesh_cache.pop("/World/object/target_object", None)
        return update(mg, *a, **k)
    planner._update_world_collision = fresh_object_mesh
    return planner, process_robot_cfg(load_robot_cfg(str(rob["yml"]))), obj


def plan(p, out: Path) -> dict:
    """The whole plan of the problem -> the plan.json dict (series + `ok`)."""
    import torch
    from automoma.utils.math_utils import get_open_ee_pose
    from curobo.types.math import Pose
    from curobo.wrap.reacher.motion_gen import MotionGenConfig
    seed = int(p.planner["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    load = MotionGenConfig.load_from_robot_config                       # AutoMoMa passes no seeds: inject them
    seeds = dict(ik_seed=1531 + seed, graph_seed=1531 + seed)
    MotionGenConfig.load_from_robot_config = staticmethod(lambda *a, **k: load(*a, **seeds, **k))
    times, t0 = {}, time.perf_counter()
    work = out / "plan"
    planner, robot_cfg, obj = setup(p, work)
    times["setup_s"] = time.perf_counter() - t0
    obj_pose = Pose.from_list(planner.object_pose)
    planner.motion_gen = planner.init_motion_gen(robot_cfg, fixed_base=False, collision_role="ik", enable_collision=True)
    q_start = [*p.start_base_obj, *p.start_arm]
    seed_cfg = [*p.start_base_obj, *p.ready]
    reasons, pulled = [], None
    t1 = time.perf_counter()
    for gid, g in obj["grasps"].items():
        def target(angle):
            return get_open_ee_pose(obj_pose, Pose.from_list(g["pose_obj"]), planner.object_urdf, p.handle.collision,
                                    {p.joint: angle}, {p.joint: 0.0})
        try:
            rows, q_g = plan_pull(planner, p, g, gid, target(0.0), target(p.goal), seed_cfg)
            drive, dt = drive_in(planner, robot_cfg, p.joint, q_start, rows[0], p.planner["timing"]["standoff_m"], p.world_pose)
        except NoPlan as e:
            reasons.append(str(e))
            continue
        pulled = (gid, rows, drive, dt)
        break
    times["plan_s"] = time.perf_counter() - t1
    if pulled is None:
        raise NoPlan("; ".join(reasons))
    gid, rows, drive, dt = pulled
    world = rows.copy()
    world[:, :3] = frames.base_to_world(rows[:, :3], p.world_pose)
    vel = planner.motion_gen.kinematics.kinematics_config.joint_limits.velocity[1].cpu().numpy()[:10]
    timing = p.planner["timing"]
    pull_s = series.pull_duration(world, vel, timing["pull_slowdown"], timing["pull_min_s"])
    s = series.build(drive, dt, world, (p.cfg["gripper_open"], p.cfg["gripper_squeeze"]), timing, pull_s)
    report = {"grasp": gid, "grasp_along_m": p.grasp_along[gid], "tried": reasons, "times": times, "pull_s": pull_s,
              "drive_in_s": dt * (len(drive) - 1) * timing["drive_slowdown"], "events": s["events"],
              "door_goal": p.goal, "start_world": p.start_base, "grasp_base_world": world[0, :3].tolist(),
              "pull_end_base_world": world[-1, :3].tolist()}
    (work / "plan_report.json").write_text(json.dumps(report, indent=1))
    return {"ok": True, "expert": EXPERT, "report": report, "rate_hz": s["rate_hz"], "base_accel": timing["base_accel"],
            "base": np.round(s["base"], 5).tolist(), "arm": np.round(s["arm"], 5).tolist(),
            "hand": np.round(s["hand"], 5).tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="datagen.plugins.automoma.plan")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--out", required=True, help="the scene folder: plan.json goes here")
    a = ap.parse_args(argv)
    out = Path(a.out)
    try:
        from revolve_common import specs
        result = plan(problem.build(a.instance, a.task, a.robot, specs.REPO_SPECS.parent), out)
        rc = 0
    except (NoPlan, problem.NotSupported) as e:
        result, rc = {"ok": False, "reason": str(e)}, 3
    (out / "plan.json").write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k in ("ok", "reason", "report")}, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
