"""Plan whole-body trajectories with AutoMoMa's planner (plan step only) for a task: open the fridge door
(open_fridge_door) or the drawer (open_drawer), with the robot of $REVOLVE_ROBOT (the experiment's).

  python plan.py <out_dir> task=.. [scene=.. generator=..]   (entry.py plan, in revolve-datagen)
  env ATTEMPTS, SEED override the generator's attempts (N) and seed; CENTRE_SEED the starts block's
  centre seed (default SEED).
Ported from the earlier pipeline's plan.py: specs via specs_bridge (was its plugin loader),
the robot's gz scripts from the simulator component; the centre seed from CENTRE_SEED instead of the
legacy round's manifest.json.

Uses AutoMoMa's own classes in its order: CuroboPlanner.setup_env (scene USD -> voxel
ESDF), plan_ik at the closed and the goal joint position per grasp, cluster_ik, plan_traj on
the AKR of that grasp, filter_traj, PlanningPipeline._convert_to_12d. One difference: an
attempt is one trajopt start-goal problem, and a run has exactly N of them, split evenly
over the grasps with IK; per grasp the closest start/goal cluster pairs (joint-space L2,
distinct starts first) are taken, instead of AutoMoMa's all-pairs product.

Writes <out>/assets/ (gen_assets.py), <out>/automoma/traj_data_train.pt (AutoMoMa's format,
all N attempts, traj_success marks the planned ones), <out>/handoff.npz + handoff.json
(torch-free, see handoff.json "format") and <out>/plan_report.json.

With a round's starts block (env ROUND_STARTS, start_design.py) the attempts start from clustered
starts instead: attempt a plans the grasp of its cluster from its drawn start. Its start IK holds
the base at the start (base joints locked, READY arm seed). Its goal IK is seeded with the start
moved rigidly with the handle (yaw clamped into the window); without a solution it takes the
nearest clustered goal IK of the grasp. The base yaw window of the block bounds the IK and the
trajopt.
handoff.json then holds a "starts" record and each attempt its "cluster".
A `mode: reach` block (start_design.ReachSpec) plans the seed's one sampled grasp from its sampled base
(attempt 0) and small jitters of it (the other attempts), the same pinned start IK; the "starts" record
holds the sample and each attempt's d, turn and base.
A mint round's plan dir holds plan_request.json (start_design.read_plan_request): the stop rule's
cluster of each attempt and centre seed of each cluster. The attempts then start from those clusters
and centres, and the "starts" record adds "centre_seeds".
"""
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import frames
import gen_assets
import specs_bridge as plugins
import start_design

DEFAULTS = {"generator": "automoma"}
AUTOMOMA_ROBOT_COLS = 10    # AutoMoMa's _convert_to_12d splits the state at this column: robot | object


def now():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return time.perf_counter()


def pick_pairs(S, G, k):
    """k (start, goal) index pairs, closest first, distinct starts before repeats."""
    order = torch.cdist(S, G).flatten().argsort().tolist()
    pairs, starts = [], set()
    for idx in order:
        s, g = divmod(idx, G.shape[0])
        if s not in starts and len(pairs) < k:
            pairs.append((s, g))
            starts.add(s)
    for idx in order:
        if len(pairs) < k and divmod(idx, G.shape[0]) not in pairs:
            pairs.append(divmod(idx, G.shape[0]))
    return pairs


LBFGS_FUSED_HISTORIES = (5, 6, 7, 15, 27, 31)   # cuRobo lbfgs_step_kernel.cu compiled history sizes
IK_SEEDS_STARTS = 1000  # IK seeds per attempt with a starts block (start IK: the arm joints, base locked)


def round_first_seed(out, seed):
    """The centre seed of a starts block: $CENTRE_SEED, else the run's own seed (as a legacy plan run
    outside a round, or plan run 1 of a round)."""
    return int(os.environ.get("CENTRE_SEED", seed))


class PinnedStartIK:
    """Start IK of a starts block: AutoMoMa's IK problem (its IK collision checker, collision on)
    with base_x, base_y and base_yaw locked at the start, the arm seeded with ik_seed's nominal arm
    and regularised toward it (the robot's READY). One cuRobo IKSolver; each start moves the locked base
    with update_kinematics_config, as cuRobo's MotionGen.update_locked_joints does."""

    def __init__(self, planner, robot_cfg, base, seed, arm):
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        self.planner, self.robot_cfg = planner, robot_cfg
        coll = planner._get_collision_checker(fixed_base=False, enable_collision=True, collision_role="ik")
        self.solver = IKSolver(IKSolverConfig.load_from_robot_config(
            self.config(base), None, planner.tensor_args, IK_SEEDS_STARTS, world_coll_checker=coll,
            use_cuda_graph=False, seed=seed))
        # cuRobo cuts the L-BFGS history to d_opt - 1 for a short chain (5 arm joints: 4), and its fused kernel
        # for a history without a compiled variant reads out of bounds (CUDA illegal memory access): those take
        # cuRobo's own non-fused step instead (the 7-joint arm's history is compiled, so it is unchanged).
        for opt in self.solver.solver.optimizers:
            if getattr(opt, "use_cuda_kernel", False) and getattr(opt, "history", 5) not in LBFGS_FUSED_HISTORIES:
                opt.use_cuda_kernel = False
        self.ready = planner.tensor_args.to_device(list(arm))

    def config(self, base):
        from curobo.types.robot import RobotConfig
        cfg = copy.deepcopy(self.robot_cfg)
        k = cfg["kinematics"]
        k["lock_joints"] = {**dict(zip(("base_x", "base_y", "base_yaw"), map(float, base))), **k["lock_joints"]}
        return RobotConfig.from_dict(cfg, self.planner.tensor_args)

    def solve(self, target, q_seed):
        """(start IK (base + arm) on the CPU or None, solutions found, index of the one taken) for a TCP
        target Pose and the seed (start base, arm seed). Takes the solution nearest READY that the
        whole-body IK model of plan_ik also finds feasible at the target."""
        from curobo.types.state import JointState
        self.solver.kinematics.update_kinematics_config(self.config(q_seed[:3]).kinematics.kinematics_config)
        arm = self.planner.tensor_args.to_device(list(q_seed[3:]))
        r = self.solver.solve_single(target, retract_config=self.ready[None], seed_config=arm[None, None],
                                     return_seeds=IK_SEEDS_STARTS, num_seeds=IK_SEEDS_STARTS)
        sol = r.solution.reshape(-1, self.solver.dof)[r.success.reshape(-1)]
        if not len(sol):
            return None, 0, -1
        i = int(torch.linalg.norm(sol - self.ready, dim=1).argmin())
        q = torch.cat([self.planner.tensor_args.to_device(list(q_seed[:3])), sol[i]])[None]
        full = self.planner.motion_gen.ik_solver
        ee = full.fk(q).ee_pose
        feasible = bool(full.check_constraints(JointState.from_position(q)).feasible.all())
        if not feasible or float(torch.linalg.norm(ee.position - target.position)) > 0.01:
            return None, len(sol), -1
        return q[0].cpu(), len(sol), i


def goal_seed(q_start, tcp_start, tcp_goal, yaw_window):
    """Goal IK seed (base + arm): the start base moved rigidly with the TCP from its start to its goal pose
    (planner frame, 4x4), the yaw clamped into the window, the start arm."""
    M = tcp_goal @ np.linalg.inv(tcp_start)
    x, y, yaw = (float(v) for v in q_start[:3])
    B = M @ frames.object_tf((x, y, 0.0, yaw))
    lo, hi = yaw_window
    yaw_g = min(max(yaw + float(np.arctan2(M[1, 0], M[0, 0])), lo + 1e-3), hi - 1e-3)
    return torch.cat([torch.tensor([B[0, 3], B[1, 3], yaw_g], dtype=q_start.dtype), q_start[3:]])


def seeded_goal_ik(planner, target, seed):
    """(goal IK (base + arm) on the CPU or None) for a TCP target Pose: plan_ik's IK solver regularised
    toward and seeded with `seed`, the solution nearest the seed."""
    ta = planner.tensor_args
    s = ta.to_device(seed)
    r = planner.motion_gen.ik_solver.solve_single(target, retract_config=s[None], seed_config=s[None, None],
                                                  return_seeds=IK_SEEDS_STARTS, num_seeds=IK_SEEDS_STARTS)
    sol = r.solution.reshape(-1, s.shape[0])[r.success.reshape(-1)]
    if not len(sol):
        return None
    return sol[int(torch.linalg.norm(sol - s, dim=1).argmin())].cpu()


def plan_clustered(planner, A, task, gen, robot_cfg, obj_pose, starts, yaw_window, seed, arm):
    """rows, results (one per start, in attempt order), ik report: the plan step of a starts block.
    arm: the nominal arm of the starts (the robot's READY)."""
    from automoma.core.types import TrajResult
    from automoma.utils.file_utils import load_robot_cfg, process_robot_cfg
    from automoma.utils.math_utils import get_open_ee_pose, stack_iks_angle
    from curobo.types.math import Pose
    jn, goal = task["joint"], float(task["goal_angle"])
    ax, grasps = A["object"]["grasp_axis"], A["object"]["grasps"]
    closed = {jn: 0.0}
    used = sorted({s.grasp_id for s in starts})

    def target(gid, angle):
        return get_open_ee_pose(obj_pose, Pose.from_list(grasps[gid]["pose_obj"]), planner.object_urdf,
                                task["handle"], {jn: angle}, closed)

    # Start IK at the closed door, one per start (plan_ik's IK MotionGen, made here as plan_ik does).
    planner.motion_gen = planner.init_motion_gen(robot_cfg, fixed_base=False, collision_role="ik",
                                                 enable_collision=True)
    planner._update_world_collision(planner.motion_gen, closed, True, include_object_mesh=True)
    tcp = {(gid, angle): target(gid, angle) for gid in used for angle in (0.0, goal)}
    pin = PinnedStartIK(planner, robot_cfg, start_design.ik_seed(starts[0], arm)[:3], 1531 + seed, arm)
    q_start, found, pick = {}, {}, {}
    for st in starts:
        q_start[st.attempt_id], found[st.attempt_id], pick[st.attempt_id] = pin.solve(
            tcp[st.grasp_id, 0.0], start_design.ik_seed(st, arm))
    del pin
    planner.free_cuda_cache()
    # Goal IK at the goal angle, one per start: seeded with the start moved rigidly with the handle.
    # A start without a seeded goal takes the nearest of the grasp's clustered goal IK (plan_ik).
    planner._update_world_collision(planner.motion_gen, {jn: goal}, True, include_object_mesh=True)
    q_goal, goal_src, goal_pick = {}, {}, {}
    ik = {gid: dict(goal_raw=None, goal_clustered=None) for gid in used}
    for st in starts:
        a = st.attempt_id
        if q_start[a] is not None:
            mats = [tcp[st.grasp_id, ang].get_matrix()[0].cpu().numpy().astype(np.float64) for ang in (0.0, goal)]
            q_goal[a] = seeded_goal_ik(planner, tcp[st.grasp_id, goal],
                                       goal_seed(q_start[a], *mats, yaw_window).tolist())
            goal_src[a] = "seeded" if q_goal[a] is not None else "clustered"
    for gid in used:
        mine = [s.attempt_id for s in starts if s.grasp_id == gid]
        if any(goal_src.get(a) == "clustered" for a in mine):
            r = planner.plan_ik(torch.tensor(tcp[gid, goal].to_list()), robot_cfg,
                                plan_cfg={"joint_cfg": {jn: goal}, "enable_collision": True})
            G = r.iks[planner.cluster_ik(r)].cpu() if len(r) else r.iks.cpu()
            planner.free_cuda_cache()
            ik[gid].update(goal_raw=len(r), goal_clustered=len(G))
            for a in mine:
                if goal_src.get(a) == "clustered" and len(G):
                    goal_pick[a] = int(torch.linalg.norm(G - q_start[a], dim=1).argmin())
                    q_goal[a] = G[goal_pick[a]]
        ik[gid].update(start_attempts=len(mine), start_found=sum(q_start[a] is not None for a in mine),
                       goal_seeded=sum(goal_src.get(a) == "seeded" for a in mine))
        print(f"grasp {gid}: start IK {ik[gid]['start_found']} of {ik[gid]['start_attempts']} starts, seeded goal IK "
              f"{ik[gid]['goal_seeded']}, clustered goal IK {ik[gid]['goal_raw']}")

    steps = gen["planner"]["traj"]["trajopt_tsteps"]
    rows, results = [None] * len(starts), [None] * len(starts)
    for gid in used:
        idx = [s.attempt_id for s in starts if s.grasp_id == gid and q_goal.get(s.attempt_id) is not None]
        akr = process_robot_cfg(load_robot_cfg(grasps[gid]["akr_yml"]))
        if idx:
            tr = planner.plan_traj(stack_iks_angle(torch.stack([q_start[a] for a in idx]), 0.0),
                                   stack_iks_angle(torch.stack([q_goal[a] for a in idx]), -goal), akr,
                                   plan_cfg={"expand_to_pairs": False, "enable_collision": True,
                                             "batch_size": min(len(idx), gen["planner"]["traj"]["batch_size"])})
        for st in (s for s in starts if s.grasp_id == gid):
            a = st.attempt_id
            if a in idx:
                one = tr[idx.index(a):idx.index(a) + 1]
                ok = bool(one.success.any())
                passed = ok and planner.filter_traj(one, akr).num_samples > 0
            else:       # no start IK or no goal IK: a failed attempt at its drawn start, READY arm
                q = stack_iks_angle(torch.tensor([start_design.ik_seed(st, arm)], dtype=torch.float32), 0.0)
                one = TrajResult(q, q, q[:, None].repeat(1, steps, 1), torch.tensor([False]))
                ok = passed = False
            rows[a] = dict(attempt_id=a, grasp_id=gid, cluster=st.cluster, **{f"grasp_{ax}": grasps[gid][ax]},
                           start_ik_cluster=pick[a], goal_ik_cluster=goal_pick.get(a, -1), goal_ik=goal_src.get(a),
                           trajopt_success=ok, filter_pass=passed, success=passed, start_ik_solutions=found[a])
            results[a] = TrajResult(one.start_states, one.goal_states, one.trajectories, torch.tensor([passed]))
        planner.free_cuda_cache()
    return rows, results, ik


def convert_12d(pipe, merged, nq):
    """AutoMoMa's _convert_to_12d for nq = 3 + arm DoF robot columns: it splits robot / object at column
    AUTOMOMA_ROBOT_COLS, so the robot is zero-padded to that width before and the pad dropped after (zero
    columns for a 7-DoF arm)."""
    from automoma.core.types import TrajResult
    pad = AUTOMOMA_ROBOT_COLS - nq
    if pad < 0:
        raise SystemExit(f"{nq} robot joints (3 + arm DoF) > AutoMoMa's {AUTOMOMA_ROBOT_COLS} robot columns: an arm "
                         f"above {AUTOMOMA_ROBOT_COLS - 3} DoF needs an AutoMoMa patch (ThirdParty/patches/automoma)")

    def wide(t):
        return torch.cat([t[..., :nq], t.new_zeros((*t.shape[:-1], pad)), t[..., nq:]], -1)
    conv = pipe._convert_to_12d(TrajResult(wide(merged.start_states), wide(merged.goal_states),
                                           wide(merged.trajectories), merged.success))
    for key in ("start_robot", "goal_robot", "traj_robot"):
        conv[key] = torch.cat([conv[key][..., :nq], conv[key][..., AUTOMOMA_ROBOT_COLS:]], -1)
    return conv


def main(out, sel):
    out = Path(out).resolve()          # AutoMoMa joins relative urdf paths onto /opt/automoma
    if (out / "handoff.json").exists():
        raise SystemExit(f"{out} already holds a run")
    if "task" not in sel:
        raise SystemExit("plan.py: task=<task> is required")
    r = plugins.task_spec(sel["task"])           # robot = $REVOLVE_ROBOT (the experiment's), no default
    names = {"robot": r.robot.name, "scene": r.scene.name, "task": sel["task"], **DEFAULTS, **sel}
    if names["robot"] != r.robot.name:
        raise SystemExit(f"plan.py: robot={names['robot']} but $REVOLVE_ROBOT names {r.robot.name}")
    rd, sc, task, gen = (plugins.load(k, names[k]) for k in ("robot", "scene", "task", "generator"))
    rb = gen_assets.gz_robot(rd)
    arm = tuple(float(x) for x in rb.ready)      # the starts' nominal arm (generator starts nominal_arm: ready)
    attempts = int(os.environ.get("ATTEMPTS", gen["attempts"]))
    seed = int(os.environ.get("SEED", gen["seed"]))
    spec = start_design.starts_from_env(os.environ)
    reach = isinstance(spec, start_design.ReachSpec)
    if reach:
        if (out / start_design.REQUEST_FILE).exists():
            raise SystemExit(f"{start_design.REQUEST_FILE} needs a clustered starts block, not mode reach")
        if attempts < 1:
            raise SystemExit("no attempts: ATTEMPTS is 0")
        # One grasp: the seed's sampled point along the task's grasp axis (the attempts only jitter the base).
        sample = start_design.reach_sample(seed, spec)
        along, = (k for k in "xyz" if k in task["grasp"])
        task["grasp"] = {**task["grasp"], along: [sample.grasp]}
    elif spec is not None:
        if task["name"] not in rb.planner.BOX_CENTRE_OBJ:
            raise SystemExit(f"starts: no box centre for task {task['name']} in robots/{rb.implementation}/planner.py "
                             "BOX_CENTRE_OBJ (or drop the starts block for this robot in generator.yaml)")
        box = rb.planner.BOX_CENTRE_OBJ[task["name"]]
        centre_seed = round_first_seed(out, seed)
        request = start_design.read_plan_request(out / start_design.REQUEST_FILE, seed, attempts, spec)
        starts = (start_design.plan_starts(centre_seed, seed, attempts, spec, box) if request is None
                  else start_design.request_starts(*request, seed, spec, box))
        if not starts:
            raise SystemExit("no attempts: ATTEMPTS is 0")
    elif (out / start_design.REQUEST_FILE).exists():
        raise SystemExit(f"{start_design.REQUEST_FILE} needs a starts block (ROUND_STARTS)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    from curobo.wrap.reacher.motion_gen import MotionGenConfig
    load = MotionGenConfig.load_from_robot_config     # AutoMoMa passes no seeds: inject them
    seeds = dict(ik_seed=1531 + seed, graph_seed=1531 + seed)
    MotionGenConfig.load_from_robot_config = staticmethod(lambda *a, **k: load(*a, **seeds, **k))

    smi_log = open("/tmp/smi.csv", "w+")
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                            "-lms", "100"], stdout=smi_log)
    T = {}
    t0 = now()
    A = gen_assets.build(rd, sc, task, out / "assets", None if spec is None else start_design.yaw_window_rad(spec))
    # cuRobo's fixed-base trajopt config (AutoMoMa's choice) with the generator's running pose
    # weights: the AKR end link must stay put at every waypoint, and the default rotation weight
    # lets it turn by up to 20 deg mid-pull on the PartNet door (B4).
    if gen.get("trajopt_pose_weight"):
        from curobo.util_file import get_task_configs_path, join_path, load_yaml
        tcfg = load_yaml(join_path(get_task_configs_path(), "gradient_trajopt_fixbase.yml"))
        tcfg["cost"]["pose_cfg"]["weight"] = gen["trajopt_pose_weight"]
        tf = out / "assets" / "gradient_trajopt_fixbase.yml"
        tf.write_text(yaml.safe_dump(tcfg, sort_keys=False))
        gen["planner"]["traj"]["gradient_trajopt_file"] = str(tf)       # absolute: cuRobo uses it as is
    T["assets_s"] = now() - t0
    if reach:
        # The base that puts the sampled grasp at reach d (facing the object), turned about the arm's first joint.
        grasp_obj = np.array(A["object"]["grasps"][0]["pose_obj"][:3])
        pivot = rb.first_joint_xy()
        starts = start_design.reach_starts(
            seed, attempts, spec, lambda d: rb.base_for_reach(grasp_obj, start_design.FACING_YAW_OBJ, d=d), pivot)

    from automoma.core.types import TrajResult
    from automoma.planning.pipeline import PlanningPipeline
    from automoma.utils.file_utils import load_object_from_metadata, load_robot_cfg, process_robot_cfg
    from automoma.utils.math_utils import get_open_ee_pose, stack_iks_angle
    from curobo.types.math import Pose

    t = now()
    pipe = PlanningPipeline({"planner": gen["planner"]})
    planner = pipe.planner
    scene_cfg = {"path": str(A["scene"]["usd"]), "pose": [0, 0, 0, 1, 0, 0, 0],
                 "metadata_path": str(A["scene"]["metadata"])}
    obj_cfg = load_object_from_metadata(scene_cfg["metadata_path"], {
        "path": str(A["object"]["urdf"]), "asset_type": task["object"], "asset_id": task["object"]})
    planner.setup_env(scene_cfg, obj_cfg)
    T["setup_env_s"] = now() - t
    # AutoMoMa re-meshes the object per door angle under one name, and cuRobo's warp cache
    # keys meshes by name, so every IK would see the first (closed) door. Drop the entry first.
    update = planner._update_world_collision

    def fresh_object_mesh(mg, *a, **k):
        mg.world_coll_checker._wp_mesh_cache.pop("/World/object/target_object", None)
        return update(mg, *a, **k)
    planner._update_world_collision = fresh_object_mesh
    robot_cfg = process_robot_cfg(load_robot_cfg(str(A["robot"]["yml"])))
    obj_pose = Pose.from_list(planner.object_pose)
    jn, goal = task["joint"], float(task["goal_angle"])
    unit = {"revolute": "rad", "prismatic": "m"}[A["object"]["joint_type"]]
    ax = A["object"]["grasp_axis"]
    closed = {jn: 0.0}

    if spec is not None:
        t = now()
        rows, results, ik_report = plan_clustered(planner, A, task, gen, robot_cfg, obj_pose, starts,
                                                  start_design.yaw_window_rad(spec), seed, arm)
        quota = {g: sum(s.grasp_id == g for s in starts) for g in sorted({s.grasp_id for s in starts})}
        T["ik_traj_s"] = now() - t
    else:
        # IK at the closed door (start) and at the goal angle, per grasp; clustered as AutoMoMa.
        t = now()
        ik = {}
        for gid, g in A["object"]["grasps"].items():
            sets = []
            for angle in (0.0, goal):
                target = get_open_ee_pose(obj_pose, Pose.from_list(g["pose_obj"]), planner.object_urdf,
                                          task["handle"], {jn: angle}, closed)
                r = planner.plan_ik(torch.tensor(target.to_list()), robot_cfg,
                                    plan_cfg={"joint_cfg": {jn: angle}, "enable_collision": True})
                sets.append((len(r), r.iks[planner.cluster_ik(r)].cpu() if len(r) else r.iks.cpu()))
                planner.free_cuda_cache()
            ik[gid] = dict(start=sets[0][1], goal=sets[1][1], raw=(sets[0][0], sets[1][0]))
            print(f"grasp {gid}: IK start {sets[0][0]} -> {len(sets[0][1])}, goal {sets[1][0]} -> {len(sets[1][1])}")
        T["ik_s"] = now() - t

        ok = [g for g in ik if len(ik[g]["start"]) and len(ik[g]["goal"])]
        quota = {g: attempts // len(ok) + (i < attempts % len(ok)) for i, g in enumerate(ok)} if ok else {}
        t = now()
        rows, results = [], []
        for gid in ok:
            pairs = pick_pairs(ik[gid]["start"], ik[gid]["goal"], quota[gid])
            S = stack_iks_angle(ik[gid]["start"][[s for s, _ in pairs]], 0.0)
            G = stack_iks_angle(ik[gid]["goal"][[g for _, g in pairs]], -goal)   # AKR: -(door angle)
            akr = process_robot_cfg(load_robot_cfg(A["object"]["grasps"][gid]["akr_yml"]))
            tr = planner.plan_traj(S, G, akr, plan_cfg={"expand_to_pairs": False, "batch_size": len(pairs),
                                                        "enable_collision": True})
            for i, (s, g) in enumerate(pairs):
                one = tr[i:i + 1]
                passed = bool(one.success.any()) and planner.filter_traj(one, akr).num_samples > 0
                rows.append(dict(attempt_id=len(rows), grasp_id=gid, **{f"grasp_{ax}": A["object"]["grasps"][gid][ax]},
                                 start_ik_cluster=s, goal_ik_cluster=g,
                                 trajopt_success=bool(one.success.any()), filter_pass=passed, success=passed))
                results.append(TrajResult(one.start_states, one.goal_states, one.trajectories,
                                          torch.tensor([passed])))
            planner.free_cuda_cache()
        ik_report = {g: {"start_raw": v["raw"][0], "start_clustered": len(v["start"]), "goal_raw": v["raw"][1],
                         "goal_clustered": len(v["goal"])} for g, v in ik.items()}
        T["traj_s"] = now() - t
    if not results:
        raise SystemExit("no attempts: no grasp has both start and goal IK")

    # AutoMoMa's own 12-D conversion and .pt writer (all attempts; traj_success = planned).
    nq = 3 + rb.dof                                   # base + arm joints
    conv = convert_12d(pipe, TrajResult.cat(results), nq)
    (out / "automoma").mkdir(parents=True, exist_ok=True)
    pt = out / "automoma" / "traj_data_train.pt"
    pipe.planning_io.save_converted(conv, str(pt))

    # Planner TCP (cuRobo FK of the robot's TCP link) and base, both to the world frame.
    obj = task["world_pose"]
    traj = conv["traj_robot"].numpy().astype(np.float64)            # (A, 36, nq + 2) object frame
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
                 base_yaw_world=[float(w[:, 2].min()), float(w[:, 2].max())],
                 base_start_world=w[0, :3].tolist(), base_end_world=w[-1, :3].tolist(),
                 **{f"door_final_{unit}": float(door[r["attempt_id"], -1])})
    smi.terminate()
    smi.wait()
    smi_log.seek(0)
    smi_mib = [int(x) for x in smi_log.read().split() if x.isdigit()]

    jnames = robot_cfg["kinematics"]["cspace"]["joint_names"]
    success = np.array([r["success"] for r in rows])
    np.savez_compressed(
        out / "handoff.npz", joint_names=np.array(jnames), traj_robot_obj=traj, traj_robot_world=world,
        door_angle=door, tcp_world=tcp_world, success=success,
        attempt_id=np.arange(len(rows)), grasp_id=np.array([r["grasp_id"] for r in rows]))
    handoff = {
        "format": {
            "files": {"handoff.npz": "numpy arrays below, one row per attempt",
                      "automoma/traj_data_train.pt": "AutoMoMa 12-D .pt, same rows, planner frame"},
            "joint_names": jnames,
            "traj_robot_obj": f"(A, 36, {nq + 2}) AutoMoMa traj_robot (base, arm, 2 gripper columns): base in "
                              f"the object ({task['object']}) frame",
            "traj_robot_world": f"(A, 36, {nq + 2}) same with base_x, base_y, base_yaw in the world = gz joint "
                                "values; frames.base_to_world(traj_robot_obj[..., :3], object_world_pose)",
            "door_angle": f"(A, 36) {unit}, gz {task['object']}::{jn} position: 0 closed, positive opens "
                          f"(axis {[round(v, 6) for v in A['object']['joint_axis_model']]} in the model frame). "
                          f"= AutoMoMa traj_obj; AutoMoMa's AKR state holds -door_angle ({jn}_akr)",
            "tcp_world": f"(A, 36, 4, 4) planner FK of {rd['tcp']['link']}, world frame",
            "steps": "rows 0-3 = waypoint 0 while the gripper closes (fingers 0.04 -> 0), rows 4-35 = "
                     "the 32 trajopt waypoints; no time step (cuRobo optimize_dt, not kept by AutoMoMa)",
            "quaternions": "wxyz wherever a 7-D pose appears (cuRobo, AutoMoMa)",
            "base_yaw": "not wrapped",
            "success": "trajopt success and AutoMoMa's FK filter; replay only success rows"},
        "run": out.name, "plugins": names, "seed": seed, "curobo_seeds": seeds,
        "torch_seed": seed, "numpy_seed": seed, "attempts": len(rows), "planned": int(success.sum()),
        "object_world_pose": obj, "goal_angle": goal,
        "grasps": {g: {k: v[k] for k in (ax, "pose_obj", "T_handle_grasp")}
                   for g, v in A["object"]["grasps"].items()},
        "attempts_detail": rows}
    if reach:
        handoff["starts"] = {
            "mode": "reach", "facing_yaw_obj": start_design.FACING_YAW_OBJ, "spec": vars(spec),
            "sample": {f"grasp_{ax}": sample.grasp, "d_m": sample.d, "dyaw_deg": float(np.degrees(sample.dyaw))},
            "grasp_obj": grasp_obj.tolist(), "first_joint_xy": pivot.tolist(),
            "attempts": [{"attempt_id": st.attempt_id, "d_m": st.dx, "dyaw_deg": float(np.degrees(st.dyaw)),
                          "base_obj": [st.x, st.y, st.yaw]} for st in starts]}
    elif spec is not None:
        centres = (start_design.cluster_centres(centre_seed, spec, box) if request is None
                   else start_design.request_centres(request[1], spec, box))
        handoff["starts"] = {
            "centre_seed": centre_seed, **({} if request is None else {"centre_seeds": list(request[1])}),
            "facing_yaw_obj": start_design.FACING_YAW_OBJ, "box_centre_obj": list(box),
            "centres_obj": [list(c) for c in centres]}
    (out / "handoff.json").write_text(json.dumps(handoff, indent=1))
    T["total_s"] = now() - t0
    # planned ends 5 deg / 5 cm beyond the replay's hold threshold (60 deg, 15 cm)
    margin = {"rad": ("planned_door_final_ge_65deg", np.radians(65)), "m": ("planned_drawer_final_ge_20cm", 0.20)}[unit]
    report = {
        "attempts": len(rows), "planned": int(success.sum()), "success_rate": float(success.mean()),
        "trajopt_success": int(sum(r["trajopt_success"] for r in rows)),
        margin[0]: int(sum(r["success"] and r[f"door_final_{unit}"] >= margin[1] for r in rows)),
        "ik_per_grasp": ik_report,
        "attempts_per_grasp": quota, "times": T, "seed": seed, "curobo_seeds": seeds,
        "vram_baseline_mib": smi_mib[0] if smi_mib else None,
        "vram_peak_mib": max(smi_mib) if smi_mib else None,
        "torch_max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch_max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "scene_cuboids": len(A["scene"]["cuboids"]), "door_spheres": A["object"]["door_spheres"],
        "planner": gen["planner"], "trajopt_pose_weight": gen.get("trajopt_pose_weight")}
    (out / "plan_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "planner"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], dict(a.split("=") for a in sys.argv[2:]))
