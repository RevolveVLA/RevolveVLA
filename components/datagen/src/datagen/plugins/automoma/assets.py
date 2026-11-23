"""Planner assets, generated per plan run from the specs (the scene instance, task, robot) into <run>/assets/:

  robot/<robot>.urdf, <robot>.yml        planner URDF + cuRobo robot cfg: world -> base_x, base_y, base_yaw -> base_link
                                         (= base_footprint) -> the arm (cuRobo's franka chain) -> the TCP, base + arm spheres
  object/<object>.urdf                   the task's scene model as boxes, root link "world" = the model frame
  object/<robot>_<object>_grasp_NNNN.{urdf,yml}   AKR: robot -> grasp -> handle -> door -> reversed joint -> object root ->
                                         anchor (ee_link), one per grasp
  scene/<scene>.usda, metadata.json      the static world as cuboids + the object's world pose

Everything but the cuRobo-backed pieces (the franka chain, the USD writer) is plain python; those import lazily and run in
the datagen image only."""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml

from datagen.plugins.moveit.geometry import grasp_pose, quat, tcp_frame

from . import frames
from .model import Model, tf

ARM_LINKS = [f"panda_link{n}" for n in range(8)] + ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
FINGERS = ["panda_finger_joint1", "panda_finger_joint2"]


def fmt(v):
    return " ".join(f"{float(x):.9g}" for x in v)


def rpy(T):
    """Fixed-axis roll, pitch, yaw of a rotation (R = Rz Ry Rx)."""
    R = T[:3, :3]
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    if abs(R[2, 0]) > 1 - 1e-9:
        return 0.0, pitch, math.atan2(-R[0, 1], R[1, 1])
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


def pose7(T):
    """x y z qw qx qy qz."""
    x, y, z, w = quat(T[:3, :3])
    return [*map(float, T[:3, 3]), w, x, y, z]


def origin(T):
    return f'<origin xyz="{fmt(T[:3, 3])}" rpy="{fmt(rpy(T))}"/>'


def joint(name, typ, parent, child, T=np.eye(4), axis=None, lim=None):
    s = f'<joint name="{name}" type="{typ}"><parent link="{parent}"/><child link="{child}"/>{origin(T)}'
    if axis is not None:
        s += f'<axis xyz="{fmt(axis)}"/>'
    if lim is not None:
        s += f'<limit lower="{lim[0]:.9g}" upper="{lim[1]:.9g}" velocity="{lim[2]:.9g}" effort="1000"/>'
    return s + "</joint>"


def link(name, boxes=()):
    geo = "".join(f'<{k}>{origin(T)}<geometry><box size="{fmt(s)}"/></geometry></{k}>' for _, T, s in boxes
                  for k in ("visual", "collision"))
    return f'<link name="{name}">{geo}</link>'


def urdf(name, links, joints):
    return f'<?xml version="1.0"?>\n<robot name="{name}">\n' + "\n".join(links + joints) + "\n</robot>\n"


# ---------------------------------------------------------------------- robot
def margin(j, cfg) -> float:
    """How far inside its limits the planner keeps an arm joint (not the fingers) [rad]."""
    return cfg["joint_margin"] if j.get("name").startswith("panda_joint") else 0.0


def franka_chain(cfg, tcp: str):
    """dict(links, joints, spheres, ignore, buffer): the Panda from cuRobo's own description (kinematics, limits, spheres), the
    TCP `cfg['tcp_z']` along the hand z. cuRobo's content is the datagen image's."""
    import xml.etree.ElementTree as ET

    from curobo.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml
    root = ET.parse(join_path(get_assets_path(), "robot/franka_description/franka_panda.urdf")).getroot()
    joints = []
    for j in root.findall("joint"):
        if j.get("name") not in [f"panda_joint{n}" for n in range(1, 9)] + ["panda_hand_joint", *FINGERS]:
            continue
        o = j.find("origin")
        T = tf([float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy").split()])
        lim, ax = j.find("limit"), j.find("axis")
        joints.append(joint(j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"), T,
                            None if j.get("type") == "fixed" else [float(v) for v in ax.get("xyz").split()],
                            None if j.get("type") == "fixed" else
                            (float(lim.get("lower")) + margin(j, cfg), float(lim.get("upper")) - margin(j, cfg),
                             float(lim.get("velocity")))))
    joints.append(joint(f"{tcp}_joint", "fixed", "panda_hand", tcp, tf((0, 0, cfg["tcp_z"]))))
    kin = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]["kinematics"]
    spheres = load_yaml(join_path(get_robot_configs_path(), kin["collision_spheres"]))["collision_spheres"]
    drop = "attached_object"
    return dict(links=[*ARM_LINKS, "panda_link8", tcp], joints=joints, spheres={n: spheres[n] for n in ARM_LINKS},
                ignore={k: [x for x in v if x != drop] for k, v in kin["self_collision_ignore"].items() if k != drop},
                buffer={k: v for k, v in kin["self_collision_buffer"].items() if k != drop})


def robot(p, chain, out: Path):
    """Planner URDF parts and cuRobo kinematics cfg of the problem's robot: the base chain (world -> base_x, base_y,
    base_yaw -> base_link -> the arm's root link at the mount) + `chain` (franka_chain), the base spheres, the gripper locked
    open. Base limits in the object frame: the image of the world box, the yaw window around the start's."""
    c = p.cfg
    to_obj = np.linalg.inv(frames.object_tf(p.world_pose))
    lim = c["base"]["limits"]
    pts = np.array([(to_obj @ [sx * lim, sy * lim, 0, 1])[:2] for sx in (-1, 1) for sy in (-1, 1)])
    lo, hi = pts.min(0), pts.max(0)
    yaw0 = p.start_base_obj[2]
    yaw = (yaw0 - p.yaw_window, yaw0 + p.yaw_window)
    lock = {n: c["gripper_open"] for n in FINGERS}
    joints = [joint("base_x", "prismatic", "world", "base_x_link", axis=(1, 0, 0), lim=(lo[0], hi[0], c["base"]["v_max"])),
              joint("base_y", "prismatic", "base_x_link", "base_y_link", axis=(0, 1, 0),
                    lim=(lo[1], hi[1], c["base"]["v_max"])),
              joint("base_yaw", "revolute", "base_y_link", "base_link", axis=(0, 0, 1), lim=(*yaw, c["base"]["w_max"])),
              joint("base_mount", "fixed", "base_link", "panda_link0", tf(c["mount_xyz"]))]
    joints += chain["joints"]
    links = [link(n) for n in ["world", "base_x_link", "base_y_link", "base_link", *chain["links"]]]
    spheres = dict(chain["spheres"])
    b = c["base_spheres"]
    spheres["base_link"] = [{"center": [x, y, z], "radius": b["radius"]} for x in b["x"] for y in b["y"] for z in b["z"]]
    ignore, buffer = dict(chain["ignore"]), dict(chain["buffer"])
    ignore["base_link"], buffer["base_link"] = list(c["base_self_collision_ignore"]), 0.0
    arm = [f"panda_joint{n}" for n in range(1, 8)]
    names = ["base_x", "base_y", "base_yaw", *arm, *lock]
    n = len(names)
    kcfg = {"use_usd_kinematics": False, "urdf_path": "", "asset_root_path": "", "base_link": "world", "ee_link": p.tcp,
            "collision_link_names": ["base_link", *chain["spheres"]], "collision_spheres": spheres,
            "collision_sphere_buffer": c["collision_sphere_buffer"], "use_global_cumul": True,
            "self_collision_ignore": ignore, "self_collision_buffer": buffer, "lock_joints": lock,
            "cspace": {"joint_names": names, "retract_config": [*p.start_base_obj, *p.ready, *lock.values()],
                       "null_space_weight": [1] * n, "cspace_distance_weight": [1] * n,
                       "max_acceleration": c["max_acceleration"], "max_jerk": c["max_jerk"]}}
    d = out / "robot"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{p.robot}.urdf"
    path.write_text(urdf(p.robot, links, joints))
    kcfg["urdf_path"] = str(path)
    yml = d / f"{p.robot}.yml"
    yml.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": kcfg}}, sort_keys=False))
    return dict(links=links, joints=joints, kinematics=kcfg, urdf=path, yml=yml)


# ---------------------------------------------------------------------- object
def box_distance(point, T, size) -> float:
    """Signed distance of a point to a box (4x4 pose, size)."""
    d = np.abs((np.linalg.inv(T) @ [*point, 1.0])[:3]) - np.asarray(size) / 2
    return float(np.linalg.norm(np.maximum(d, 0)) + min(max(d), 0))


def door_spheres(m: Model, handle: str, cfg: dict):
    """Spheres on the moving link in its frame: a grid over its panel (the largest collision box that is not the handle) without
    the ones that touch the rest of the object at the closed door (the planner's world holds those boxes), and a row along the
    handle bar. cuRobo's self-collision kernel takes at most 216 spheres."""
    door = m.joint.child
    _, T_p, size = max((b for b in m.boxes[door] if b[0] != handle), key=lambda b: np.prod(b[2]))
    size = np.asarray(size)
    thin = int(np.argmin(size))
    grid = [np.linspace(-s / 2, s / 2, int(np.ceil(s / cfg["step"] - 1e-9)) + 1) for i, s in enumerate(size) if i != thin]
    r = max(float(size[thin]) / 2, cfg["radius"])
    rest = [(m.links[ln] @ T, s) for ln, boxes in m.boxes.items() if ln != door for _, T, s in boxes]
    out = []
    for a in grid[0]:
        for b in grid[1]:
            c = np.zeros(3)
            c[[i for i in range(3) if i != thin]] = a, b
            centre = (T_p @ [*c, 1.0])[:3]
            if all(box_distance((m.links[door] @ [*centre, 1.0])[:3], T, s) - r >= cfg["clearance"] for T, s in rest):
                out.append({"center": centre.tolist(), "radius": r})
    _, T_h, hsize = next(b for b in m.boxes[door] if b[0] == handle)
    ax = int(np.argmax(hsize))
    r = float(np.linalg.norm(np.delete(hsize, ax))) / 2
    end = hsize[ax] / 2 - r / 2
    for t in np.linspace(-end, end, int(np.ceil(2 * end / r)) + 1):
        out.append({"center": (T_h @ [*(t * np.eye(3)[ax]), 1])[:3].tolist(), "radius": r})
    return out


def task_object(p, m: Model, rob, out: Path):
    """Object URDF, the grasps and one AKR URDF + yml per grasp. The task joint moves the link that holds the handle."""
    j, door, parent = m.joint, m.joint.child, m.joint.parent
    T_lh = next(T for n, T, _ in m.boxes[door] if n == p.handle.collision)
    H0 = m.links[door] @ T_lh
    links, joints = [link("world"), link(p.handle.collision)], []
    for ln, boxes in m.boxes.items():
        links.append(link(ln, boxes))
        name, par = m.parent_of[ln]
        rel = np.linalg.inv(m.links[par]) @ m.links[ln] if par != "world" else m.links[ln]
        if par == "world":
            joints.append(joint(f"{ln}_fixed", "fixed", "world", ln, m.links[ln]))
        elif ln == door:
            joints.append(joint(j.name, j.type, par, ln, rel, j.axis, (j.lower, j.upper, 1.0)))
        else:                                          # other doors stay closed
            joints.append(joint(f"{name}_closed", "fixed", par, ln, rel))
    joints.append(joint(f"{p.handle.collision}_frame", "fixed", door, p.handle.collision, T_lh))
    d = out / "object"
    d.mkdir(parents=True, exist_ok=True)
    obj_urdf = d / f"{p.object}.urdf"
    obj_urdf.write_text(urdf(p.object, links, joints))

    spheres = door_spheres(m, p.handle.collision, p.door_spheres)
    root, grasps = f"{p.object}_root", {}
    R = tcp_frame(H0, int(np.argmax(next(s for n, _, s in m.boxes[door] if n == p.handle.collision))))
    for gid, at in enumerate(p.grasp_along):
        G = grasp_pose(H0, R, (0.0, 0.0, at))
        T_hg = np.linalg.inv(H0) @ G
        akr_j = [joint("akr_grasp", "fixed", rob["kinematics"]["ee_link"], p.handle.collision, np.linalg.inv(T_hg)),
                 joint("akr_handle", "fixed", p.handle.collision, door, np.linalg.inv(T_lh)),
                 # value = -(joint position): AutoMoMa's "AKR mechanical convention"
                 joint(f"{j.name}_akr", j.type, door, f"{j.name}_akr_link", axis=j.axis, lim=(-j.upper, -j.lower, 1.0)),
                 joint(f"{j.name}_akr_parent", "fixed", f"{j.name}_akr_link", parent,
                       np.linalg.inv(np.linalg.inv(m.links[parent]) @ m.links[door])),
                 joint(f"akr_{parent}_up", "fixed", parent, root, np.linalg.inv(m.links[parent]))]
        # AKR end link: a body-fixed frame halfway along the grasp's path from closed to the goal. trajopt holds its position
        # tightly but its rotation loosely, so a short lever keeps the TCP on the real handle.
        Gg = m.door_pose(p.goal) @ T_lh @ T_hg
        anchor = f"{p.object}_akr_anchor"
        akr_j.append(joint("akr_anchor", "fixed", root, anchor, tf((G[:3, 3] + Gg[:3, 3]) / 2)))
        akr_l = [link(x) for x in [p.handle.collision, door, f"{j.name}_akr_link", parent, root, anchor]]
        stem = f"{p.robot}_{p.object}_grasp_{gid:04d}"
        akr_urdf = d / f"{stem}.urdf"
        akr_urdf.write_text(urdf(stem, rob["links"] + akr_l, rob["joints"] + akr_j))
        k = copy.deepcopy(rob["kinematics"])
        k.update(urdf_path=str(akr_urdf), ee_link=anchor)
        k["collision_link_names"].append(door)
        k["collision_spheres"][door] = spheres
        k["self_collision_ignore"][door] = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        k["self_collision_buffer"][door] = 0.0
        cs = k["cspace"]
        at_ = 3 + p.arm_dof                         # after the base and arm joints, before the gripper locks
        cs["joint_names"].insert(at_, f"{j.name}_akr")
        for key, v in (("retract_config", 0.0), ("null_space_weight", 1), ("cspace_distance_weight", 1)):
            cs[key].insert(at_, v)
        yml = d / f"{stem}.yml"
        yml.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": k}}, sort_keys=False))
        grasps[gid] = dict(along=at, pose_obj=pose7(G), T_handle_grasp=T_hg.tolist(), akr_yml=str(yml))
    return dict(urdf=obj_urdf, grasps=grasps, door_spheres=len(spheres), handle_zero=H0.tolist(),
                joint_type=j.type)


# ----------------------------------------------------------------------- scene
def scene(p, out: Path):
    """The static world as cuboids -> USD (/World/scene/obstacles/*) and AutoMoMa's metadata.json with the object's pose."""
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.util.usd_helper import UsdHelper
    fl = p.floor
    cubes = [*p.boxes, ("floor", tf((0, 0, -fl["thickness"] / 2)), [*fl["size"], fl["thickness"]])]
    d = out / "scene"
    d.mkdir(parents=True, exist_ok=True)
    usd = d / f"{p.scene}.usda"
    world = WorldConfig(cuboid=[Cuboid(name=re.sub(r"\W", "_", n), pose=pose7(T), dims=[float(x) for x in s]) for n, T, s in cubes])
    u = UsdHelper()
    u.create_stage(str(usd), base_frame="/World")
    u.add_subroot("/World", "scene")
    u.add_world_to_stage(world, base_frame="/World/scene")
    u.write_stage_to_file(str(usd))
    x, y, z, yaw = p.world_pose
    meta = d / "metadata.json"
    meta.write_text(json.dumps({"static_objects": {p.object: {
        "name": p.object, "asset_type": p.object, "asset_id": p.object, "position": [x, y, z], "rotation": [0.0, 0.0, yaw],
        "dimensions": [0.0, 0.0, 0.0]}}}, indent=1))
    return dict(usd=usd, metadata=meta, cuboids=len(cubes))
