"""Planner assets, generated at plan time from the plugin data and the gz sources (the
generator is the source of truth; outputs are not committed):

  robot/<robot>.urdf, <robot>.yml      planner URDF + cuRobo robot cfg (ee_link panda_tcp)
  object/<object>.urdf                 the task's gz object (fridge, cabinet) as a URDF, root link
                                       "world" = model frame
  object/grasp/NNNN.npy                TCP pose in the object frame, joint closed (x y z qw qx qy qz)
  object/<robot>_<object>_grasp_NNNN.{urdf,yml}   AKR: robot -> grasp -> handle -> door or drawer ->
                                       reversed joint (hinge or slide) -> object root -> anchor
                                       (ee_link), one per grasp
  scene/<scene>.usda, metadata.json    static world as cuboids + the object's world pose

Runs in the revolve-datagen image (cuRobo for the sphere files and the USD writer). Ported from
the earlier pipeline's gen_assets.py: specs via specs_bridge; gz_scripts = the simulator component.
"""
import copy
import json
import math
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import frames
import specs_bridge as plugins
import sdf


def fmt(v):
    return " ".join(f"{float(x):.9g}" for x in v)


def pose7(T):
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [*map(float, T[:3, 3]), float(w), float(x), float(y), float(z)]


def origin(T):
    return f'<origin xyz="{fmt(T[:3, 3])}" rpy="{fmt(Rotation.from_matrix(T[:3, :3]).as_euler("xyz"))}"/>'


def joint(name, typ, parent, child, T=np.eye(4), axis=None, lim=None):
    s = f'<joint name="{name}" type="{typ}"><parent link="{parent}"/><child link="{child}"/>{origin(T)}'
    if axis is not None:
        s += f'<axis xyz="{fmt(axis)}"/>'
    if lim is not None:
        s += f'<limit lower="{lim[0]:.9g}" upper="{lim[1]:.9g}" velocity="{lim[2]:.9g}" effort="1000"/>'
    return s + "</joint>"


def link(name, boxes=()):
    geo = "".join(f'<{k}>{origin(T)}<geometry><box size="{fmt(s)}"/></geometry></{k}>'
                  for _, T, s in boxes for k in ("visual", "collision"))
    return f'<link name="{name}">{geo}</link>'


def urdf(name, links, joints):
    return f'<?xml version="1.0"?>\n<robot name="{name}">\n' + "\n".join(links + joints) + "\n</robot>\n"


def gz_modules(rd):
    sys.path.insert(0, str(plugins.path(rd["gz_scripts"])))
    import mobile_panda as mp
    import panda_plan as pp
    return mp, pp


# ---------------------------------------------------------------------- robot
def robot(rd, task, out, yaw_window=None):
    """Planner URDF parts and cuRobo kinematics cfg for the mobile Panda. yaw_window: base yaw
    (low, high) in the object frame instead of the task's (a round's starts block)."""
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    mp, pp = gz_modules(rd)
    fd = plugins.path(rd["franka_description"]) / "robots/fer"
    kin = yaml.safe_load((fd / "kinematics.yaml").read_text())
    lim = yaml.safe_load((fd / "joint_limits.yaml").read_text())

    # Base limits in the object-centred planner frame: the image of gz's +-3 m box (exact
    # for an object yaw that is a multiple of pi/2), yaw from the task window.
    gx, gy, gyaw = rd["gz_base_limits"]
    to_obj = np.linalg.inv(frames.object_tf(task["world_pose"]))
    pts = np.array([(to_obj @ [sx * gx, sy * gy, 0, 1])[:2] for sx in (-1, 1) for sy in (-1, 1)])
    lo, hi = pts.min(0), pts.max(0)
    yaw = task["base_yaw_window"] if yaw_window is None else yaw_window
    assert all(abs(y + task["world_pose"][3]) <= gyaw for y in yaw), "yaw window beyond gz limits"

    joints = [
        joint("base_x", "prismatic", "world", "base_x_link", axis=(1, 0, 0),
              lim=(lo[0], hi[0], mp.BASE_V_MAX)),
        joint("base_y", "prismatic", "base_x_link", "base_y_link", axis=(0, 1, 0),
              lim=(lo[1], hi[1], mp.BASE_V_MAX)),
        joint("base_yaw", "revolute", "base_y_link", "base_link", axis=(0, 0, 1),
              lim=(yaw[0], yaw[1], mp.BASE_W_MAX)),
        joint("base_mount", "fixed", "base_link", "panda_link0", sdf.tf((mp.MOUNT_X, 0, mp.MOUNT_Z)))]
    for n in range(1, 8):
        k = kin[f"joint{n}"]["kinematic"]
        lj = lim[f"joint{n}"].get("limit", lim[f"joint{n}"])
        joints.append(joint(f"panda_joint{n}", "revolute", f"panda_link{n - 1}", f"panda_link{n}",
                            sdf.tf((k["x"], k["y"], k["z"]), (k["roll"], k["pitch"], k["yaw"])),
                            (0, 0, 1), (lj["lower"], lj["upper"], lj["velocity"])))
    joints += [
        joint("panda_hand_joint", "fixed", "panda_link7", "panda_hand",
              sdf.tf((0, 0, kin["joint8"]["kinematic"]["z"]), (0, 0, pp.HAND_YAW))),
        joint("panda_tcp_joint", "fixed", "panda_hand", "panda_tcp", sdf.tf((0, 0, pp.TCP_Z)))]
    for side, fyaw in (("left", 0.0), ("right", math.pi)):      # as gen_panda_sdf.finger_link
        joints.append(joint(f"panda_finger_joint_{side}", "prismatic", "panda_hand",
                            f"panda_{side}finger", sdf.tf((0, 0, 0.0584), (0, 0, fyaw)), (0, 1, 0),
                            (mp.GRIP_CLOSED, mp.GRIP_OPEN, 0.2)))
    arm = [f"panda_link{n}" for n in range(8)] + ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
    links = [link(n) for n in ["world", "base_x_link", "base_y_link", "base_link", *arm, "panda_tcp"]]

    franka = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]["kinematics"]
    spheres = load_yaml(join_path(get_robot_configs_path(), franka["collision_spheres"]))["collision_spheres"]
    spheres = {n: spheres[n] for n in arm}
    for s in spheres["panda_rightfinger"]:      # gz's right finger frame is yawed by pi
        s["center"] = [-s["center"][0], -s["center"][1], s["center"][2]]
    b = rd["base_spheres"]
    spheres["base_link"] = [{"center": [x, y, z], "radius": b["radius"]}
                            for x in b["x"] for y in b["y"] for z in b["z"]]
    drop = "attached_object"
    ignore = {k: [x for x in v if x != drop] for k, v in franka["self_collision_ignore"].items()}
    ignore["base_link"] = rd["base_self_collision_ignore"]
    buffer = {k: v for k, v in franka["self_collision_buffer"].items() if k != drop}
    buffer["base_link"] = 0.0
    # Regeneration only (plan_from_state sets base_corner_spheres): a column of spheres at each chassis
    # corner on a fixed link of its own that collides with the world but not with the robot or the
    # task object, so cuRobo's self-collision pairs stay within its kernel's 32 x 512 limit; the base
    # keeps a margin to the task object through its self-collision buffer instead.
    c = rd.get("base_corner_spheres")
    coll = ["base_link"] + arm
    if c:
        joints.append(joint("base_corners_joint", "fixed", "base_link", "base_corners"))
        links.append(link("base_corners"))
        spheres["base_corners"] = [{"center": [sx * c["xy"], sy * c["xy"], z], "radius": c["radius"]}
                                   for sx in (-1, 1) for sy in (-1, 1) for z in c["z"]]
        ignore["base_corners"] = ["base_link"] + arm
        buffer["base_corners"] = 0.0
        buffer["base_link"] = rd.get("base_self_collision_buffer", 0.0)
        coll.append("base_corners")
    names = ["base_x", "base_y", "base_yaw"] + [f"panda_joint{n}" for n in range(1, 8)] + list(rd["lock_joints"])
    kcfg = {
        "use_usd_kinematics": False, "urdf_path": "", "asset_root_path": "",
        "base_link": "world", "ee_link": rd["ee_link"],
        "collision_link_names": coll, "collision_spheres": spheres,
        "collision_sphere_buffer": rd["collision_sphere_buffer"], "use_global_cumul": True,
        "self_collision_ignore": ignore, "self_collision_buffer": buffer,
        "lock_joints": rd["lock_joints"],
        "cspace": {"joint_names": names,
                   "retract_config": [*task["base_retract"], *map(float, pp.READY),
                                      *rd["lock_joints"].values()],
                   "null_space_weight": rd["null_space_weight"],
                   "cspace_distance_weight": rd["cspace_distance_weight"],
                   "max_acceleration": rd["max_acceleration"], "max_jerk": rd["max_jerk"]}}
    d = out / "robot"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rd['name']}.urdf"
    path.write_text(urdf(rd["name"], links, joints))
    kcfg["urdf_path"] = str(path)
    yml = d / f"{rd['name']}.yml"
    yml.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": kcfg}}, sort_keys=False))
    return dict(links=links, joints=joints, kinematics=kcfg, urdf=path, yml=yml)


# ---------------------------------------------------------------------- object
def task_object(rd, task, rob, out):
    """Object URDF, grasp poses and one AKR URDF + yml per grasp. The task joint (revolute or
    prismatic) moves the link that holds the handle ("door": the fridge door or the drawer)."""
    mp, _ = gz_modules(rd)
    m = sdf.Model(plugins.path(task["gz_model"]) / "model.sdf")
    jn = task["joint"]
    j = m.joints[jn]
    door, parent = j["child"], j["parent"]
    assert np.allclose(m.zero(jn), m.zero(door)), "door link frame must be the hinge frame"
    h_link, T_lh = next((ln, T) for ln, bs in m.boxes.items() for n, T, _ in bs if n == task["handle"])
    assert h_link == door

    # Object URDF: "world" is the model frame (AutoMoMa reads get_transform(handle, "world")).
    links, joints = [link("world"), link(task["handle"])], []
    for ln in m.boxes:
        pj = m.joints[m.parent_joint[ln]]
        links.append(link(ln, m.boxes[ln]))
        if pj["parent"] == "world":
            joints.append(joint(f"{ln}_fixed", "fixed", "world", ln, m.zero(ln)))
        elif m.parent_joint[ln] == jn:
            joints.append(joint(jn, j["type"], pj["parent"], ln,
                                np.linalg.inv(m.zero(pj["parent"])) @ m.zero(jn), pj["axis"],
                                (pj["lower"], pj["upper"], 1.0)))
        else:                                  # other doors stay closed
            joints.append(joint(f"{m.parent_joint[ln]}_closed", "fixed", pj["parent"], ln,
                                np.linalg.inv(m.zero(pj["parent"])) @ m.zero(ln)))
    joints.append(joint(f"{task['handle']}_frame", "fixed", door, task["handle"], T_lh))
    d = out / "object"
    (d / "grasp").mkdir(parents=True, exist_ok=True)
    obj_urdf = d / f"{task['object']}.urdf"
    obj_urdf.write_text(urdf(task["object"], links, joints))

    # Grasps: as mobile_panda_skills.grasp_fridge_handle, in the model frame (front = +x), then
    # the hand rolled roll_deg about its z (-90: the fingers close vertically across a horizontal
    # bar, as grasp_drawer_bar). The TCP positions run along the handle: the grasp key x, y or z.
    H0 = m.pose(door, {}) @ T_lh
    n = np.array([1.0, 0.0, 0.0])
    R = mp.hand_rot(-n, sdf.rot((0, 0, 1), math.atan2(-n[1], -n[0])) @ [0.0, 1.0, 0.0])
    R = R @ sdf.rot((0, 0, 1), math.radians(task["grasp"]["roll_deg"]))
    along, = (k for k in "xyz" if k in task["grasp"])
    # Door spheres: the plugin's grid in the model frame (door closed), in the door link frame.
    ds, L = task["door_spheres"], np.linalg.inv(m.zero(door))
    grid = [np.linspace(a, b, int(np.ceil(abs(b - a) / ds["step"] - 1e-9)) + 1) for a, b in (ds["y"], ds["z"])]
    door_spheres = [{"center": (L @ [ds["x"], y, z, 1.0])[:3].tolist(), "radius": ds["radius"]}
                    for y in grid[0] for z in grid[1]]
    # A row along the handle box (the forearm can cross it above or below the grasp), radius
    # the half-diagonal of its cross-section so the spheres bound the box, spaced <= r apart.
    # cuRobo's fast self-collision kernel takes at most 32 x 512 sphere pairs (warns and falls
    # back to a kernel that fails to launch above ~160 spheres), so every door sphere counts.
    hsize = next(s for n, _, s in m.boxes[door] if n == task["handle"])
    ax = int(np.argmax(hsize))
    r = float(np.linalg.norm(np.delete(hsize, ax))) / 2
    end = hsize[ax] / 2 - r / 2
    for t in np.linspace(-end, end, int(np.ceil(2 * end / r)) + 1):
        door_spheres.append({"center": (T_lh @ [*(t * np.eye(3)[ax]), 1])[:3].tolist(), "radius": r})
    # Reversed chain from the hinge's parent link back to the model root.
    up, ln = [], parent
    while ln != "world":
        pj = m.joints[m.parent_joint[ln]]
        up.append((ln, pj["parent"]))
        ln = pj["parent"]
    root = f"{task['object']}_root"
    spots = []
    for at in task["grasp"][along]:
        G = np.eye(4)
        pos = H0[:3, 3].copy()
        pos["xyz".index(along)] = at
        G[:3, :3], G[:3, 3] = R, pos - task["grasp"]["depth"] * n
        spots.append((at, G))
    # Grasps measured in a saved state (plan_from_state.py, a policy's grasp): the TCP in the door
    # link frame, appended after the task's grasps.
    for T in task["grasp"].get("measured_tcp_in_link", []):
        G = m.pose(door, {}) @ np.asarray(T, float)
        spots.append((float(G["xyz".index(along), 3]), G))
    grasps = {}
    for gid, (at, G) in enumerate(spots):
        np.save(d / "grasp" / f"{gid:04d}.npy", np.array(pose7(G)))
        T_hg = np.linalg.inv(H0) @ G
        akr_j = [joint("akr_grasp", "fixed", "panda_tcp", task["handle"], np.linalg.inv(T_hg)),
                 joint("akr_handle", "fixed", task["handle"], door, np.linalg.inv(T_lh)),
                 # value = -(gz joint position): AutoMoMa's "AKR mechanical convention"
                 joint(f"{jn}_akr", j["type"], door, f"{jn}_akr_link", axis=j["axis"],
                       lim=(-j["upper"], -j["lower"], 1.0)),
                 joint(f"{jn}_akr_parent", "fixed", f"{jn}_akr_link", parent,
                       np.linalg.inv(np.linalg.inv(m.zero(parent)) @ m.zero(jn)))]
        for c, p in up:
            p_name = root if p == "world" else p
            Tcp = np.linalg.inv(m.zero(c)) @ (m.zero(p) if p != "world" else np.eye(4))
            akr_j.append(joint(f"akr_{c}_up", "fixed", c, p_name, Tcp))
        # AKR end link: a body-fixed frame halfway along the grasp's path from closed to the
        # goal (the chord of a hinge's arc, or the slide itself). trajopt holds its position
        # tightly but its rotation loosely (weights 50000 vs 2000), so a short lever keeps the
        # TCP on the real handle; the fridge root on the floor is ~1.3 m away and gave up to
        # 6.7 cm of drift.
        Gg = m.pose(door, {jn: float(task["goal_angle"])}) @ T_lh @ T_hg
        anchor = f"{task['object']}_akr_anchor"
        akr_j.append(joint("akr_anchor", "fixed", root, anchor, sdf.tf((G[:3, 3] + Gg[:3, 3]) / 2)))
        akr_l = [link(x) for x in [task["handle"], door, f"{jn}_akr_link", *[c for c, _ in up], root, anchor]]
        stem = f"{rd['name']}_{task['object']}_grasp_{gid:04d}"
        akr_urdf = d / f"{stem}.urdf"
        akr_urdf.write_text(urdf(stem, rob["links"] + akr_l, rob["joints"] + akr_j))
        k = copy.deepcopy(rob["kinematics"])
        k.update(urdf_path=str(akr_urdf), ee_link=anchor)
        k["collision_link_names"].append(door)
        k["collision_spheres"][door] = door_spheres
        k["self_collision_ignore"][door] = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        if "base_corners" in k["self_collision_ignore"]:
            k["self_collision_ignore"]["base_corners"].append(door)
        k["self_collision_buffer"][door] = 0.0
        cs = k["cspace"]
        cs["joint_names"].insert(10, f"{jn}_akr")
        for key, v in (("retract_config", 0.0), ("null_space_weight", 1), ("cspace_distance_weight", 1)):
            cs[key].insert(10, v)
        yml = d / f"{stem}.yml"
        yml.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": k}}, sort_keys=False))
        grasps[gid] = dict({along: at}, pose_obj=pose7(G), T_handle_grasp=T_hg.tolist(), akr_yml=str(yml))
    return dict(urdf=obj_urdf, grasp_dir=d / "grasp", grasps=grasps, grasp_axis=along, handle_zero=H0.tolist(),
                door_spheres=len(door_spheres), joint_type=j["type"],
                joint_axis_model=(m.zero(jn)[:3, :3] @ j["axis"]).tolist())


# ----------------------------------------------------------------------- scene
def scene(sc, task, out):
    """Static world cuboids from the gz world file -> USD (/World/scene/obstacles/*) and
    AutoMoMa's metadata.json with the object's pose."""
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.util.usd_helper import UsdHelper
    cubes = []
    for name, uri, T, inline in sdf.world_entities(plugins.path(sc["world"])):
        if name in sc["skip"]:
            continue
        if inline is not None or name in sc["detailed"] or name == task["object"]:
            m = inline or sdf.Model(plugins.model_dir(sc["assets"], uri) / "model.sdf")
            assert not m.meshes, f"{name}: mesh collisions {m.meshes} (box-only scene, unseen)"
            moving = {m.joints[task["joint"]]["child"]} if name == task["object"] else set()
            cubes += [(f"{name}_{c}", T @ m.pose(ln, {}) @ Tc, size)
                      for ln, bs in m.boxes.items() if ln not in moving for c, Tc, size in bs]
        else:
            man = json.loads((plugins.model_dir(sc["assets"], uri) / "manifest.json").read_text())
            lo, hi = np.array(man["aabb_model_frame"])
            cubes.append((f"{name}_aabb", T @ sdf.tf((lo + hi) / 2), hi - lo))
    fl = sc["floor"]
    cubes.append(("floor", sdf.tf((0, 0, -fl["thickness"] / 2)), [*fl["size"], fl["thickness"]]))
    d = out / "scene"
    d.mkdir(parents=True, exist_ok=True)
    usd = d / f"{sc['name']}.usda"
    world = WorldConfig(cuboid=[Cuboid(name=n, pose=pose7(T), dims=[float(x) for x in s])
                                for n, T, s in cubes])
    u = UsdHelper()
    u.create_stage(str(usd), base_frame="/World")
    u.add_subroot("/World", "scene")
    u.add_world_to_stage(world, base_frame="/World/scene")
    u.write_stage_to_file(str(usd))
    x, y, z, yaw = task["world_pose"]
    meta = d / "metadata.json"
    meta.write_text(json.dumps({"static_objects": {task["object"]: {
        "name": task["object"], "asset_type": task["object"], "asset_id": task["object"],
        "position": [x, y, z], "rotation": [0.0, 0.0, yaw],
        # 0: AutoMoMa clears nothing from the ESDF (see generator expanded_dims)
        "dimensions": [0.0, 0.0, 0.0]}}}, indent=1))
    return dict(usd=usd, metadata=meta, cuboids=[(n, pose7(T), [float(v) for v in s]) for n, T, s in cubes])


def build(rd, sc, task, out, yaw_window=None):
    rob = robot(rd, task, out, yaw_window)
    return dict(robot=rob, object=task_object(rd, task, rob, out), scene=scene(sc, task, out))
