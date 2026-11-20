"""Static FK check of a plan run (numpy only, runs in either image).

  python fk_check.py <run_dir> [n]        writes <run_dir>/fk_check.json, exit 0 = pass

At every waypoint, in the world frame (TCP gate on the first n planned trajectories, or all
when fewer plan, at least 1; drift reported for all planned ones):
- planner TCP (cuRobo FK of robot.yaml tcp.link, handoff tcp_world) vs the Gazebo TCP frame, computed
  from the SDF Gazebo loads (robot gz_sdf, posed by its <include> in the world file) at
  base_x, base_y, base_yaw = frames.base_to_world(planner base). Pass: < 2 cm and < 10 deg.
- grasp drift (reported, not gated): Gazebo TCP vs the grasp on the gz object's handle
  at the planned joint position (door angle or drawer slide), which checks the sign of the
  hand-off's door_angle.
- joint limits of the planner URDF vs the SDF.
"""
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import frames
import specs_bridge as plugins
import sdf


def main(run, n=None):
    run = Path(run)
    h = json.loads((run / "handoff.json").read_text())
    z = np.load(run / "handoff.npz")
    rd = plugins.load("robot", h["plugins"]["robot"])
    task = plugins.load("task", h["plugins"]["task"])
    world = plugins.load("scene", h["plugins"]["scene"])["world"]
    n = n or plugins.load("generator", h["plugins"]["generator"])["fk_check_trajectories"]
    robot = sdf.Model(plugins.path(rd["gz_sdf"]))
    obj = sdf.Model(plugins.path(task["gz_model"]) / "model.sdf")
    placed = {name: T for name, _, T, _ in sdf.world_entities(plugins.path(world))}
    # the scene's robot include is the placeholder `robot` (the simulator names it for the robot at run time)
    M_robot, M_obj = placed["robot"], placed[task["object"]]
    door = obj.joints[task["joint"]]["child"]
    T_lh = next(T for nm, T, _ in obj.boxes[door] if nm == task["handle"])
    names = [str(x) for x in z["joint_names"]]

    per, worst_p, worst_a = [], 0.0, 0.0
    for a in np.flatnonzero(z["success"]):
        T_hg = np.array(h["grasps"][str(int(z["grasp_id"][a]))]["T_handle_grasp"])
        ep, ea, dp, da = [], [], [], []
        for k in range(z["traj_robot_world"].shape[1]):
            T_gz = M_robot @ robot.pose(rd["tcp"]["link"], dict(zip(names, z["traj_robot_world"][a, k])))
            T_pl = z["tcp_world"][a, k]
            ep.append(np.linalg.norm(T_gz[:3, 3] - T_pl[:3, 3]))
            ea.append(math.degrees(sdf.angle_between(T_gz[:3, :3], T_pl[:3, :3])))
            G = M_obj @ obj.pose(door, {task["joint"]: z["door_angle"][a, k]}) @ T_lh @ T_hg
            dp.append(np.linalg.norm(T_gz[:3, 3] - G[:3, 3]))
            da.append(math.degrees(sdf.angle_between(T_gz[:3, :3], G[:3, :3])))
        per.append(dict(attempt_id=int(a), grasp_id=int(z["grasp_id"][a]), fk_gated=len(per) < n, waypoints=len(ep),
                        max_pos_err_m=float(max(ep)), max_ang_err_deg=float(max(ea)),
                        grasp_drift_max_m=float(max(dp)), grasp_drift_max_deg=float(max(da))))
        if len(per) <= n:
            worst_p, worst_a = max(worst_p, max(ep)), max(worst_a, max(ea))

    # Joint limits: planner URDF vs the SDF Gazebo loads.
    u = ET.parse(run / "assets" / "robot" / f"{rd['name']}.urdf").getroot()
    lim = {j.get("name"): j.find("limit").attrib for j in u.findall("joint") if j.find("limit") is not None}
    arm = {n: max(abs(float(lim[n][key]) - robot.joints[n][s])            # robot.yaml arm joints
                  for key, s in (("lower", "lower"), ("upper", "upper"), ("velocity", "velocity")))
           for n in rd["joints"]["arm"]}
    corners = [frames.base_to_world([x, y, 0.0], task["world_pose"])[:2]
               for x in (float(lim["base_x"]["lower"]), float(lim["base_x"]["upper"]))
               for y in (float(lim["base_y"]["lower"]), float(lim["base_y"]["upper"]))]
    world_box = [float(np.min(corners, 0)[0]), float(np.max(corners, 0)[0]),
                 float(np.min(corners, 0)[1]), float(np.max(corners, 0)[1])]
    gz_box = [robot.joints["base_x"]["lower"], robot.joints["base_x"]["upper"],
              robot.joints["base_y"]["lower"], robot.joints["base_y"]["upper"]]
    yaw_world = [float(lim["base_yaw"][k]) + task["world_pose"][3] for k in ("lower", "upper")]
    limits = dict(arm_max_abs_diff=arm, base_xy_world_box=world_box, gz_base_xy_box=gz_box,
                  base_yaw_window_world=yaw_world,
                  gz_base_yaw=[robot.joints["base_yaw"]["lower"], robot.joints["base_yaw"]["upper"]])
    limits_ok = (max(arm.values()) < 1e-6 and np.allclose(world_box, gz_box, atol=1e-6)
                 and limits["gz_base_yaw"][0] <= yaw_world[0] and yaw_world[1] <= limits["gz_base_yaw"][1])

    ok = len(per) >= 1 and worst_p < 0.02 and worst_a < 10.0 and limits_ok
    res = dict(passed=bool(ok), trajectories=min(len(per), n), requested=n, planned=len(per),
               tol_m=0.02, tol_deg=10.0,
               max_pos_err_m=float(worst_p), max_ang_err_deg=float(worst_a),
               max_grasp_drift_m=float(max((p["grasp_drift_max_m"] for p in per), default=0)),
               max_grasp_drift_deg=float(max((p["grasp_drift_max_deg"] for p in per), default=0)),
               limits_ok=bool(limits_ok), limits=limits, per_trajectory=per,
               gz_sdf=rd["gz_sdf"], world=world)
    (run / "fk_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_trajectory", "limits")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None))
