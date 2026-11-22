#!/usr/bin/env python3
"""Turn an articulated URDF (or a Habitat *.object_config.json) into a Gazebo
SDF model that loads under the MuJoCo physics plugin.

    python3 -I urdf_to_gz.py INPUT OUT_DIR [--config cfg.yaml] [--name NAME]

INPUT is a .urdf or a Habitat/ReplicaCAD *.object_config.json (a single rigid
object; it is wrapped in a one-link URDF first). OUT_DIR/<name>/ receives
model.sdf, model.config, meshes/, manifest.json and the sanitised source URDF.

Steps: sanitise the URDF text (comma-separated vectors), drop empty root
links, apply joint overrides (live / fixed, limits, damping, friction), build
collisions (box/cylinder/sphere primitives kept; a mesh with several parts is
treated as a ready-made convex decomposition and split; a single non-convex
mesh goes through CoACD; box-like pieces become box primitives), write one
mesh file per collision (gz-physics#1162), recompute dummy inertias from the
collision volumes, convert Y-up sources to Z-up and put the lowest point at
z = 0. See README.md "Asset pipeline" and configs/ for the options.

Inputs are treated as data only: XML is parsed, meshes are read with trimesh,
nothing from the asset directory is executed.
"""
import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml

DEFAULTS = {
    "name": None,
    "description": "",
    "up_axis": "z",            # "y" for Y-up sources (Habitat / ReplicaCAD)
    "ground_align": True,      # shift so the lowest collision point is at z = 0
    "center_xy": False,        # also centre the collision AABB on x = y = 0
    "fix_to_world": False,     # add a fixed joint from the root link to world
    "density": 600.0,          # [kg/m^3] only for links without a mass
    "recompute_inertia": "auto",   # auto (dummy/missing inertia) | always | never
    "collision_mode": "auto",  # mesh collisions: auto | coacd | hull | split
    "coacd": {"threshold": 0.05, "max_convex_hull": 16, "seed": 0},
    "box_fit": 0.9,            # convex piece -> box if hull/OBB volume >= this; null = off
    "min_thickness": 0.002,    # [m] thinner pieces become boxes at least this thick
    "collision_from_visual": True,  # true: CoACD the visual of links without collisions;
                                    # "always": ignore URDF collisions, CoACD the visuals
    "default_joint_mode": "live",   # live | fixed (per-joint override in "joints")
    "joint_defaults": {
        "revolute": {"damping": 1.0, "friction": 0.5},     # N*m*s/rad, N*m
        "prismatic": {"damping": 10.0, "friction": 2.0},   # N*s/m, N
    },
    "joints": {},              # name -> {mode, position, lower, upper, damping, friction, effort}
    "joint_origin_overrides": {},  # name -> {xyz: [...], rpy: [...]}
    "links": {},               # name -> {mass, com, inertia: [ixx,iyy,izz,ixy,ixz,iyz], handles: [...],
                               #          boxes: [...], collision_rules: [...]}
    "package_paths": [],       # search roots for package:// URIs
    # Options for PartNet-Mobility-like sources (defaults change nothing):
    "scale": 1.0,              # uniform scale of the whole model (meshes, origins, prismatic limits)
    "rename": {"links": {}, "joints": {}},  # source name -> new name, applied before everything else
    "mesh_parts": "split",     # auto mode, file with several parts: split (a convex decomposition)
                               # | merge (material groups of one mesh, e.g. OBJ usemtl)
    "visual_mode": "glb",      # glb: convert non-glb visuals | copy: copy OBJ + mtl + textures
}
# links.<n>.collision_rules: [{match: [glob, ...], mode: drop|box|hull|coacd|split, merge: false,
#   name: optional}], first match wins. Globs match a URDF collision's name, its file's basename,
#   or the name of a visual of the same link using the same file. box = AABB in the link frame.
#   merge: all matched collisions are merged and processed once (e.g. one CoACD per link).
# handles / boxes: {name, xyz, rpy, size, visual, rgba, frame: link|model}; frame model = final
#   model frame (floor at z = 0, joints at their locked positions). Handles are listed in the manifest.

VEC_ATTRS = ("xyz", "rpy", "size", "scale", "rgba")


# --------------------------------------------------------------------------
# small math helpers
def rpy_to_mat(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def mat_to_rpy(R):
    sp = -R[2, 0]
    if abs(sp) < 1 - 1e-9:
        return [math.atan2(R[2, 1], R[2, 2]), math.asin(sp), math.atan2(R[1, 0], R[0, 0])]
    return [math.atan2(-R[1, 2], R[1, 1]), math.copysign(math.pi / 2, sp), 0.0]


def tf(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = rpy_to_mat(rpy)
    T[:3, 3] = xyz
    return T


def axis_angle(axis, q):
    return trimesh.transformations.rotation_matrix(q, axis)


def pose_str(T):
    vals = list(T[:3, 3]) + mat_to_rpy(T[:3, :3])
    vals = [0.0 if abs(v) < 1e-12 else v for v in vals]
    return " ".join(f"{v:.6g}" for v in vals)


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


# --------------------------------------------------------------------------
# input
def sanitize_urdf(text):
    """Fix vector attributes such as size="0.617 0.145, 0.894" (urdfdom rejects them)."""
    fixes = []

    def fix(m):
        val = m.group(2)
        new = " ".join(val.replace(",", " ").split())
        if new != val:
            fixes.append(f'{m.group(1)}="{val}" -> "{new}"')
        return f'{m.group(1)}="{new}"'

    pat = r'\b(' + "|".join(VEC_ATTRS) + r')="([^"]*)"'
    return re.sub(pat, fix, text), fixes


def object_config_to_urdf(path, name):
    """Wrap a Habitat object (render glb + collision glb + mass) in a one-link URDF."""
    with open(path) as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    render = os.path.normpath(os.path.join(base, cfg["render_asset"]))
    coll = cfg.get("collision_asset")
    coll = os.path.normpath(os.path.join(base, coll)) if coll else render
    mass = float(cfg.get("mass", 1.0))
    com = cfg.get("COM", [0, 0, 0])
    scale = cfg.get("scale", [1, 1, 1])
    s = fmt(scale)
    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}">
    <inertial>
      <origin xyz="{fmt(com)}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
    <visual><geometry><mesh filename="{render}" scale="{s}"/></geometry></visual>
    <collision><geometry><mesh filename="{coll}" scale="{s}"/></geometry></collision>
  </link>
</robot>
"""


def vec(el, attr, default):
    if el is None or el.get(attr) is None:
        return list(default)
    return [float(x) for x in el.get(attr).split()]


def parse_origin(el):
    o = el.find("origin") if el is not None else None
    return tf(vec(o, "xyz", (0, 0, 0)), vec(o, "rpy", (0, 0, 0)))


def parse_geometry(g, materials, el):
    child = list(g)[0]
    geom = {"type": child.tag}
    if child.tag == "box":
        geom["size"] = vec(child, "size", (1, 1, 1))
    elif child.tag == "cylinder":
        geom["radius"], geom["length"] = float(child.get("radius")), float(child.get("length"))
    elif child.tag == "sphere":
        geom["radius"] = float(child.get("radius"))
    elif child.tag == "mesh":
        geom["filename"] = child.get("filename")
        geom["scale"] = vec(child, "scale", (1, 1, 1))
    else:
        raise ValueError(f"unsupported geometry {child.tag}")
    mat = el.find("material")
    if mat is not None:
        c = mat.find("color")
        if c is not None:
            geom["rgba"] = vec(c, "rgba", (0.7, 0.7, 0.7, 1))
        elif mat.get("name") in materials:
            geom["rgba"] = materials[mat.get("name")]
    return geom


def parse_urdf(text):
    root = ET.fromstring(text)
    materials = {}
    for m in root.findall("material"):
        c = m.find("color")
        if c is not None:
            materials[m.get("name")] = vec(c, "rgba", (0.7, 0.7, 0.7, 1))
    links, joints = {}, {}
    for l in root.findall("link"):
        link = {"name": l.get("name"), "inertial": None, "visuals": [], "collisions": []}
        i = l.find("inertial")
        if i is not None:
            ie = i.find("inertia")
            I = np.zeros((3, 3))
            if ie is not None:
                g = lambda k: float(ie.get(k, 0))
                I = np.array([[g("ixx"), g("ixy"), g("ixz")],
                              [g("ixy"), g("iyy"), g("iyz")],
                              [g("ixz"), g("iyz"), g("izz")]])
            m = i.find("mass")
            link["inertial"] = {"mass": float(m.get("value")) if m is not None else None,
                                "T": parse_origin(i), "I": I}
        for kind in ("visual", "collision"):
            for e in l.findall(kind):
                g = e.find("geometry")
                if g is None or len(g) == 0:
                    continue
                link[kind + "s"].append({"T": parse_origin(e),
                                         "geom": parse_geometry(g, materials, e),
                                         "name": e.get("name")})
        links[link["name"]] = link
    for j in root.findall("joint"):
        lim = j.find("limit")
        dyn = j.find("dynamics")
        jt = {"name": j.get("name"), "type": j.get("type"),
              "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
              "T": parse_origin(j), "axis": vec(j.find("axis"), "xyz", (1, 0, 0)),
              "lower": float(lim.get("lower", 0)) if lim is not None else None,
              "upper": float(lim.get("upper", 0)) if lim is not None else None,
              "damping": float(dyn.get("damping", 0)) if dyn is not None else None,
              "friction": float(dyn.get("friction", 0)) if dyn is not None else None}
        joints[jt["name"]] = jt
    return root.get("name"), links, joints


def resolve_path(fn, urdf_dir, package_paths):
    if fn.startswith("file://"):
        fn = fn[len("file://"):]
    if fn.startswith("package://"):
        rest = fn[len("package://"):]
        pkg, _, sub = rest.partition("/")
        for root in package_paths:
            for cand in (os.path.join(root, pkg, sub), os.path.join(root, sub)):
                if os.path.exists(cand):
                    return cand
        cand = os.path.join(urdf_dir, sub)
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(fn)
    p = fn if os.path.isabs(fn) else os.path.join(urdf_dir, fn)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


# --------------------------------------------------------------------------
# collisions
def load_mesh_parts(path, scale):
    """All meshes in the file, node transforms applied, in the geometry frame."""
    scene = trimesh.load(path, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        m = g.copy()
        m.apply_transform(T)
        m.vertices = m.vertices * np.asarray(scale)
        parts.append(m)
    return parts


def run_coacd(mesh, opts):
    import coacd
    coacd.set_log_level("error")
    m = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    res = coacd.run_coacd(m, threshold=float(opts.get("threshold", 0.05)),
                          max_convex_hull=int(opts.get("max_convex_hull", -1)),
                          seed=int(opts.get("seed", 0)))
    return [trimesh.Trimesh(v, f) for v, f in res]


def piece_to_item(piece, T, cfg, log):
    """Convex piece -> box primitive (if box-like or degenerate) or hull mesh item."""
    try:
        hull = piece.convex_hull
    except Exception:  # qhull failure on flat input
        hull = None
    try:
        obb = piece.bounding_box_oriented
        ext = np.asarray(obb.primitive.extents, float)
        To = np.asarray(obb.primitive.transform, float)
    except Exception:  # flat piece: trimesh's hull repair needs networkx (not in the image)
        lo, hi = piece.bounds
        ext, To, hull = hi - lo, tf((lo + hi) / 2), None
    thin = ext.min() < cfg["min_thickness"] or hull is None or hull.volume <= 1e-9
    fit = cfg.get("box_fit")
    if thin or (fit and hull.volume / max(np.prod(ext), 1e-12) >= float(fit)):
        ext = np.maximum(ext, cfg["min_thickness"])
        return {"type": "box", "size": list(ext), "T": T @ To,
                "src": "degenerate->box" if thin else "box_fit"}
    hull = hull.copy()
    hull.apply_transform(T)  # mesh items are stored in the link frame
    return {"type": "mesh", "mesh": hull, "T": np.eye(4), "src": "hull"}


def mesh_collision_items(path, scale, T, mode, cfg, log):
    parts = load_mesh_parts(path, scale)
    if not parts:
        log(f"  WARNING: no triangles in {path}")
        return []
    if mode == "auto":
        if len(parts) > 1 and cfg["mesh_parts"] == "split":
            mode = "split"
        else:
            m = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
            convex = m.is_watertight and m.volume > 0 and \
                m.volume / m.convex_hull.volume > 0.97
            mode = "hull" if convex else "coacd"
    if mode == "split":
        pieces = parts
    elif mode == "hull":
        pieces = [trimesh.util.concatenate(parts)]
    elif mode == "coacd":
        pieces = run_coacd(trimesh.util.concatenate(parts), cfg["coacd"])
    else:
        raise ValueError(f"collision_mode {mode}")
    log(f"  {os.path.basename(path)}: {len(parts)} part(s), mode {mode} -> {len(pieces)} piece(s)")
    return [piece_to_item(p, T, cfg, log) for p in pieces]


def link_frame_mesh(path, scale, T):
    """All parts of a mesh file merged into one mesh in the link frame."""
    m = trimesh.util.concatenate(load_mesh_parts(path, scale))
    m.apply_transform(T)
    return m


def rule_items(meshes, mode, cfg, log, label):
    """Collision items from link-frame meshes for a collision rule."""
    if mode == "box":
        out = []
        for m in meshes:
            lo, hi = m.bounds
            out.append({"type": "box", "size": list(np.maximum(hi - lo, cfg["min_thickness"])),
                        "T": tf((lo + hi) / 2), "src": "rule box"})
        log(f"  {label}: {len(meshes)} mesh(es), mode box -> {len(out)} box(es)")
        return out
    if mode == "hull":
        pieces = meshes
    elif mode == "split":
        pieces = [p for m in meshes for p in m.split(only_watertight=False)]
    elif mode == "coacd":
        pieces = [p for m in meshes for p in run_coacd(m, cfg["coacd"])]
    else:
        raise ValueError(f"collision rule mode {mode}")
    log(f"  {label}: {len(meshes)} mesh(es), mode {mode} -> {len(pieces)} piece(s)")
    return [piece_to_item(p, np.eye(4), cfg, log) for p in pieces]


def match_rule(rules, keys):
    import fnmatch
    for r in rules:
        if any(fnmatch.fnmatchcase(k, g) for g in r["match"] for k in keys if k):
            return r
    return None


def scale_model(links, joints, s):
    """Uniform scale: mesh scales, primitive sizes, all origins and prismatic limits."""
    def sT(T):
        T = T.copy()
        T[:3, 3] *= s
        return T
    for l in links.values():
        if l["inertial"]:
            l["inertial"]["T"] = sT(l["inertial"]["T"])
            l["inertial"]["I"] = l["inertial"]["I"] * s * s
        for e in l["visuals"] + l["collisions"]:
            e["T"] = sT(e["T"])
            g = e["geom"]
            if g["type"] == "mesh":
                g["scale"] = [v * s for v in g["scale"]]
            elif g["type"] == "box":
                g["size"] = [v * s for v in g["size"]]
            else:
                g["radius"] *= s
                if "length" in g:
                    g["length"] *= s
    for j in joints.values():
        j["T"] = sT(j["T"])
        if j["type"] == "prismatic":
            for k in ("lower", "upper"):
                if j[k] is not None:
                    j[k] *= s


def rename_model(links, joints, ren):
    lm, jm = ren.get("links") or {}, ren.get("joints") or {}
    unknown = sorted(set(lm) - set(links)) + sorted(set(jm) - set(joints))
    if unknown:
        raise ValueError(f"rename names unknown links/joints {unknown}")
    new_links = {}
    for n, l in links.items():
        l["name"] = lm.get(n, n)
        new_links[l["name"]] = l
    new_joints = {}
    for n, j in joints.items():
        j["name"] = jm.get(n, n)
        j["parent"], j["child"] = lm.get(j["parent"], j["parent"]), lm.get(j["child"], j["child"])
        new_joints[j["name"]] = j
    return new_links, new_joints


def copy_obj_visual(p, src_dir, out_vis):
    """Copy an OBJ with its mtllib files and their texture maps, keeping the paths
    relative to the source directory so the references stay valid. Files outside
    src_dir are not followed. Returns the path of the copy relative to out_vis."""
    root = os.path.realpath(src_dir)

    def inside(f):
        f = os.path.realpath(f)
        return f if (f.startswith(root + os.sep) and os.path.isfile(f)) else None

    def put(f):
        rel = os.path.relpath(f, root)
        dst = os.path.join(out_vis, rel)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(f, dst)
        return rel
    src = inside(p)
    if src is None:
        raise ValueError(f"visual {p} is outside {src_dir}")
    with open(src, errors="replace") as f:
        mtls = [line.split(None, 1)[1].strip() for line in f if line.startswith("mtllib ")]
    for m in mtls:
        mp_ = inside(os.path.join(os.path.dirname(src), m))
        if mp_ is None:
            continue
        put(mp_)
        with open(mp_, errors="replace") as f:
            for line in f:
                w = line.split()
                if w and w[0].startswith("map_") and len(w) > 1:
                    tex = inside(os.path.join(os.path.dirname(mp_), w[-1]))
                    if tex:
                        put(tex)
    return put(src)


def item_points(it):
    """Points (link frame) spanning the item, for AABBs."""
    if it["type"] == "mesh":
        v = it["mesh"].vertices
    elif it["type"] == "box":
        h = np.asarray(it["size"]) / 2
        v = np.array([[sx, sy, sz] for sx in (-h[0], h[0]) for sy in (-h[1], h[1])
                      for sz in (-h[2], h[2])])
    elif it["type"] == "cylinder":
        r, l = it["radius"], it["length"]
        a = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        ring = np.c_[r * np.cos(a), r * np.sin(a)]
        v = np.r_[np.c_[ring, np.full(24, -l / 2)], np.c_[ring, np.full(24, l / 2)]]
    else:
        r = it["radius"]
        v = np.array([[sx * r, sy * r, sz * r] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return trimesh.transform_points(v, it["T"])


def item_mass_props(it):
    """(volume, centroid in link frame, inertia per unit density about centroid, link axes)."""
    R = it["T"][:3, :3]
    if it["type"] == "mesh":
        mp = it["mesh"].mass_properties  # density 1, mesh (= link) frame
        return mp["volume"], np.asarray(mp["center_mass"]), np.asarray(mp["inertia"])
    if it["type"] == "box":
        a, b, c = it["size"]
        V = a * b * c
        I = V / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    elif it["type"] == "cylinder":
        r, l = it["radius"], it["length"]
        V = math.pi * r * r * l
        I = V * np.diag([(3 * r * r + l * l) / 12, (3 * r * r + l * l) / 12, r * r / 2])
    else:
        r = it["radius"]
        V = 4 / 3 * math.pi * r ** 3
        I = V * 0.4 * r * r * np.eye(3)
    return V, it["T"][:3, 3].copy(), R @ I @ R.T


def combine_inertia(items, mass):
    """Mass distributed by volume over the items -> (com, inertia about com)."""
    props = [item_mass_props(it) for it in items]
    vol = sum(p[0] for p in props)
    rho = mass / vol
    com = sum(p[0] * p[1] for p in props) / vol
    I = np.zeros((3, 3))
    for V, c, Ic in props:
        d = c - com
        I += rho * (Ic + V * (np.dot(d, d) * np.eye(3) - np.outer(d, d)))
    return com, I, vol


def is_dummy_inertia(inertial):
    if inertial is None:
        return True
    I = inertial["I"]
    return np.allclose(I, np.eye(3)) or np.allclose(I, 0)


# --------------------------------------------------------------------------
# main conversion
def deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def convert(inp, out_root, cfg, name=None):
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    inp = os.path.abspath(inp)
    src_dir = os.path.dirname(inp)
    if inp.endswith(".object_config.json"):
        name = name or cfg["name"] or os.path.basename(inp).replace(".object_config.json", "")
        text = object_config_to_urdf(inp, name)
        log(f"object config {inp} -> one-link URDF")
    else:
        with open(inp) as f:
            text = f.read()
    text, fixes = sanitize_urdf(text)
    for fx in fixes:
        log(f"sanitised: {fx}")
    robot_name, links, joints = parse_urdf(text)
    name = name or cfg["name"] or robot_name
    out = os.path.join(out_root, name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "meshes", "collision"))
    os.makedirs(os.path.join(out, "meshes", "visual"))
    os.makedirs(os.path.join(out, "source"))
    with open(os.path.join(out, "source", "sanitized.urdf"), "w") as f:
        f.write(text)
    log(f"model {name}: {len(links)} links, {len(joints)} joints from {inp}")
    if cfg["rename"].get("links") or cfg["rename"].get("joints"):
        links, joints = rename_model(links, joints, cfg["rename"])
        log(f"renamed: {cfg['rename']}")
    scale = float(cfg["scale"])
    if scale != 1.0:
        scale_model(links, joints, scale)
        log(f"uniform scale {scale}")

    for jn, ov in cfg["joint_origin_overrides"].items():
        j = joints[jn]
        o_xyz = ov.get("xyz", list(j["T"][:3, 3]))
        o_rpy = ov.get("rpy", mat_to_rpy(j["T"][:3, :3]))
        log(f"joint {jn}: origin override xyz {o_xyz} rpy {o_rpy}")
        j["T"] = tf(o_xyz, o_rpy)

    # Drop empty root links joined to a single child by a fixed joint.
    def children_of(ln):
        return [j for j in joints.values() if j["parent"] == ln]
    child_names = {j["child"] for j in joints.values()}
    roots = [ln for ln in links if ln not in child_names]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, got {roots}")
    root, root_T = roots[0], np.eye(4)
    while True:
        l = links[root]
        kids = children_of(root)
        empty = not l["visuals"] and not l["collisions"] and \
            (l["inertial"] is None or not l["inertial"]["mass"])
        if empty and len(kids) == 1 and kids[0]["type"] == "fixed":
            log(f"dropped empty root link {root} (fixed joint {kids[0]['name']})")
            root_T = root_T @ kids[0]["T"]
            del links[root], joints[kids[0]["name"]]
            root = kids[0]["child"]
        else:
            break

    # Joint modes
    for jn, j in joints.items():
        jc = cfg["joints"].get(jn, {})
        if j["type"] in ("floating", "planar"):
            raise ValueError(f"joint {jn}: type {j['type']} not supported")
        if j["type"] == "fixed":
            j["mode"] = "fixed"
        else:
            j["mode"] = jc.get("mode", cfg["default_joint_mode"])
        j["q"] = float(jc.get("position", 0.0)) if j["mode"] == "fixed" else 0.0
    unknown = set(cfg["joints"]) - set(joints)
    if unknown:
        raise ValueError(f"config names unknown joints {sorted(unknown)}")

    # Forward kinematics (model frame before up-axis / ground alignment)
    T_up = np.eye(4)
    if cfg["up_axis"] == "y":
        T_up = tf(rpy=(math.pi / 2, 0, 0))
    elif cfg["up_axis"] != "z":
        raise ValueError("up_axis must be y or z")
    Tm = {root: T_up @ root_T}
    order = [root]
    stack = [root]
    while stack:
        p = stack.pop()
        for j in children_of(p):
            M = np.eye(4)
            ax = np.asarray(j["axis"], float)
            ax = ax / np.linalg.norm(ax)
            j["axis"] = list(ax)
            if j["q"]:
                if j["type"] in ("revolute", "continuous"):
                    M = axis_angle(ax, j["q"])
                elif j["type"] == "prismatic":
                    M[:3, 3] = ax * j["q"]
            Tm[j["child"]] = Tm[p] @ j["T"] @ M
            order.append(j["child"])
            stack.append(j["child"])

    # Collisions per link (link frame)
    coll_items, vis_items = {}, {}
    copied_visuals = {}
    deferred = []
    for ln in order:
        l = links[ln]
        lc = cfg["links"].get(ln, {})
        items = []
        force_visual = cfg["collision_from_visual"] == "always"
        rules = lc.get("collision_rules", [])
        vis_names = {}
        for v in l["visuals"]:
            if v["geom"]["type"] == "mesh" and v["name"]:
                vis_names.setdefault(v["geom"]["filename"], []).append(v["name"])
        merged = {}   # rule index -> link-frame meshes
        for c in ([] if force_visual else l["collisions"]):
            g = c["geom"]
            r = None
            if rules and g["type"] == "mesh":
                r = match_rule(rules, [c["name"], os.path.basename(g["filename"]), g["filename"]]
                               + vis_names.get(g["filename"], []))
            if r is not None:
                if r["mode"] == "drop":
                    continue
                p = resolve_path(g["filename"], src_dir, cfg["package_paths"])
                m = link_frame_mesh(p, g["scale"], c["T"])
                if r.get("merge"):
                    merged.setdefault(rules.index(r), []).append(m)
                else:
                    new = rule_items([m], r["mode"], cfg, log, os.path.basename(p))
                    for i, it in enumerate(new):
                        if r.get("name"):
                            it["cname"] = r["name"] if len(new) == 1 else f"{r['name']}_{i}"
                    items += new
                continue
            if g["type"] == "mesh":
                p = resolve_path(g["filename"], src_dir, cfg["package_paths"])
                items += mesh_collision_items(p, g["scale"], c["T"],
                                              lc.get("collision_mode", cfg["collision_mode"]),
                                              cfg, log)
            else:
                it = dict(g)
                it["T"] = c["T"]
                it["src"] = "urdf primitive"
                items.append(it)
        for ri, meshes in sorted(merged.items()):
            r = rules[ri]
            new = rule_items([trimesh.util.concatenate(meshes)], r["mode"], cfg, log,
                             f"link {ln} rule {r.get('name', ri)} (merged)")
            for i, it in enumerate(new):
                if r.get("name"):
                    it["cname"] = r["name"] if len(new) == 1 else f"{r['name']}_{i}"
            items += new
        if not items and l["visuals"] and cfg["collision_from_visual"] and not rules:
            for v in l["visuals"]:
                g = v["geom"]
                if g["type"] == "mesh":
                    p = resolve_path(g["filename"], src_dir, cfg["package_paths"])
                    log(f"  link {ln}: no collisions, CoACD of visual {os.path.basename(p)}")
                    items += mesh_collision_items(p, g["scale"], v["T"], "coacd", cfg, log)
                else:
                    it = dict(g)
                    it["T"] = v["T"]
                    it["src"] = "visual primitive"
                    items.append(it)
        handles = []
        for kind in ("handles", "boxes"):
            for h in lc.get(kind, []):
                it = {"type": "box", "size": list(h["size"]),
                      "T": tf(h.get("xyz", (0, 0, 0)), h.get("rpy", (0, 0, 0))),
                      "src": "config handle" if kind == "handles" else "config box"}
                if kind == "handles":
                    it["handle"] = h.get("name", "handle")
                elif h.get("name"):
                    it["cname"] = h["name"]
                if h.get("frame", "link") == "model":
                    deferred.append((ln, h, it))   # converted once the floor shift is known
                    continue
                items.append(it)
                handles.append((h, it))
        coll_items[ln] = items
        # Visuals: glb copied as-is (gz-common loads glb with its node transforms
        # and embedded textures); other mesh formats are converted to glb.
        vis = []
        for v in l["visuals"]:
            g = dict(v["geom"])
            if g["type"] == "mesh":
                p = resolve_path(g["filename"], src_dir, cfg["package_paths"])
                if p not in copied_visuals and cfg["visual_mode"] == "copy" and \
                        p.lower().endswith(".obj"):
                    copied_visuals[p] = copy_obj_visual(p, src_dir, os.path.join(out, "meshes", "visual"))
                elif p not in copied_visuals:
                    base = os.path.splitext(os.path.basename(p))[0]
                    fn = base + ".glb"
                    while fn in copied_visuals.values():
                        base += "_"
                        fn = base + ".glb"
                    dst = os.path.join(out, "meshes", "visual", fn)
                    if p.lower().endswith(".glb"):
                        shutil.copyfile(p, dst)
                    else:
                        trimesh.load(p, force="scene").export(dst)
                        log(f"  converted visual {os.path.basename(p)} -> {fn}")
                    copied_visuals[p] = fn
                g["uri"] = f"model://{name}/meshes/visual/{copied_visuals[p]}"
            vis.append({"T": v["T"], "geom": g})
        for h, it in handles:
            if h.get("visual", False):
                vis.append({"T": it["T"], "geom": {"type": "box", "size": h["size"],
                                                   "rgba": h.get("rgba", [0.6, 0.6, 0.62, 1])}})
        vis_items[ln] = vis

    def floor_z():
        pts = [trimesh.transform_points(item_points(it), Tm[ln])
               for ln in order for it in coll_items[ln]]
        return -np.vstack(pts)[:, 2].min() if pts and cfg["ground_align"] else 0.0
    if deferred and cfg["center_xy"]:
        raise ValueError("model-frame boxes need center_xy: false")
    if deferred:
        # model-frame boxes: final frame = Tm + floor shift (from the other collisions)
        z0 = floor_z()
        for ln, h, it in deferred:
            Tl = Tm[ln].copy()
            Tl[2, 3] += z0
            it["T"] = np.linalg.inv(Tl) @ it["T"]
            coll_items[ln].append(it)
            if h.get("visual", False):
                vis_items[ln].append({"T": it["T"], "geom": {"type": "box", "size": h["size"],
                                                             "rgba": h.get("rgba", [0.6, 0.6, 0.62, 1])}})
        if abs(floor_z() - z0) > 1e-9:
            raise ValueError("a model-frame box is the lowest collision; give it in the link frame")

    # Ground alignment from the collision AABB in the model frame
    pts = [trimesh.transform_points(item_points(it), Tm[ln])
           for ln in order for it in coll_items[ln]]
    shift = np.zeros(3)
    if pts:
        allp = np.vstack(pts)
        lo, hi = allp.min(0), allp.max(0)
        if cfg["ground_align"]:
            shift[2] = -lo[2]
        if cfg["center_xy"]:
            shift[:2] = -(lo[:2] + hi[:2]) / 2
    for ln in Tm:
        Tm[ln] = Tm[ln].copy()
        Tm[ln][:3, 3] += shift
    aabb = None
    if pts:
        aabb = [list(lo + shift), list(hi + shift)]

    # Inertials
    inertials = {}
    for ln in order:
        l = links[ln]
        lc = cfg["links"].get(ln, {})
        ine = l["inertial"]
        mass = lc.get("mass", ine["mass"] if ine and ine["mass"] else None)
        mode = cfg["recompute_inertia"]
        recompute = mode == "always" or (mode == "auto" and is_dummy_inertia(ine))
        items = coll_items[ln]
        if "inertia" in lc:
            ixx, iyy, izz, ixy, ixz, iyz = lc["inertia"]
            I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            com = np.asarray(lc.get("com", ine["T"][:3, 3] if ine else (0, 0, 0)), float)
            how = "config"
            if mass is None:
                raise ValueError(f"link {ln}: inertia override needs a mass")
        elif recompute and items:
            vol = sum(item_mass_props(it)[0] for it in items)
            if mass is None:
                mass = cfg["density"] * vol
            com, I, vol = combine_inertia(items, mass)
            how = f"from {len(items)} collision volume(s), {vol*1e3:.3g} L, {mass/vol:.0f} kg/m^3"
            if "com" in lc:
                com = np.asarray(lc["com"], float)
        elif ine is not None and not recompute:
            R = ine["T"][:3, :3]
            I = R @ ine["I"] @ R.T
            com = ine["T"][:3, 3]
            how = "urdf"
            if mass is None:
                mass = 1.0
        else:
            mass = mass or 0.01
            r = 0.05
            I = 0.4 * mass * r * r * np.eye(3)
            com = np.zeros(3)
            how = "fallback sphere"
        ev = np.linalg.eigvalsh(I)
        if ev.min() <= 0:
            I = I + np.eye(3) * (abs(ev.min()) + 1e-6)
        inertials[ln] = {"mass": float(mass), "com": np.asarray(com, float), "I": I}
        log(f"link {ln}: mass {mass:.4g} kg, inertia {how}, diag {np.round(np.diag(I), 5).tolist()}")

    # ------------------------------------------------------------------ SDF
    manifest = {"name": name, "source": inp, "config": cfg.get("_config_path"),
                "fix_to_world": cfg["fix_to_world"], "aabb_model_frame": aabb,
                "links": {}, "joints": {}, "handles": [], "collisions": 0,
                "collision_mesh_files": 0, "sanitized": fixes}
    if scale != 1.0:
        manifest["scale"] = scale
    L = []
    L.append('<?xml version="1.0"?>')
    L.append(f"<!-- Generated by asset_pipeline/urdf_to_gz.py from {os.path.basename(inp)}. Do not edit. -->")
    L.append('<sdf version="1.11">')
    L.append(f'  <model name="{name}">')
    L.append("    <self_collide>false</self_collide>")
    mesh_count = 0
    for ln in order:
        T = Tm[ln]
        ine = inertials[ln]
        I = ine["I"]
        L.append(f'    <link name="{ln}">')
        L.append(f"      <pose>{pose_str(T)}</pose>")
        L.append("      <inertial>")
        L.append(f"        <pose>{fmt(ine['com'])} 0 0 0</pose>")
        L.append(f"        <mass>{ine['mass']:.6g}</mass>")
        L.append(f"        <inertia><ixx>{I[0,0]:.6g}</ixx><ixy>{I[0,1]:.6g}</ixy><ixz>{I[0,2]:.6g}</ixz>"
                 f"<iyy>{I[1,1]:.6g}</iyy><iyz>{I[1,2]:.6g}</iyz><izz>{I[2,2]:.6g}</izz></inertia>")
        L.append("      </inertial>")
        for k, v in enumerate(vis_items[ln]):
            L.append(f'      <visual name="{ln}_visual_{k}">')
            L.append(f"        <pose>{pose_str(v['T'])}</pose>")
            L.append(f"        <geometry>{geometry_sdf(v['geom'])}</geometry>")
            if "rgba" in v["geom"]:
                c = fmt(v["geom"]["rgba"])
                L.append(f"        <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>")
            L.append("      </visual>")
        for k, it in enumerate(coll_items[ln]):
            cname = f"{ln}_{it['handle']}" if "handle" in it else \
                f"{ln}_{it['cname']}" if "cname" in it else f"{ln}_collision_{k}"
            if it["type"] == "mesh":
                fn = f"{ln}_{k:02d}.obj"
                m = it["mesh"].copy()
                m.apply_transform(it["T"])
                m.export(os.path.join(out, "meshes", "collision", fn))
                g = {"type": "mesh", "uri": f"model://{name}/meshes/collision/{fn}",
                     "scale": [1, 1, 1]}
                Tc = np.eye(4)
                mesh_count += 1
            else:
                g, Tc = it, it["T"]
            L.append(f'      <collision name="{cname}">')
            L.append(f"        <pose>{pose_str(Tc)}</pose>")
            L.append(f"        <geometry>{geometry_sdf(g)}</geometry>")
            L.append("      </collision>")
            if "handle" in it:
                manifest["handles"].append({
                    "name": cname, "link": ln, "size": it["size"],
                    "pose_link": pose_str(it["T"]),
                    "center_model_frame": list(np.round((T @ it["T"])[:3, 3], 5))})
        L.append("    </link>")
        manifest["links"][ln] = {"pose_model_frame": pose_str(T), "mass": ine["mass"],
                                 "collisions": len(coll_items[ln]),
                                 "collision_sources": sorted({it["src"] for it in coll_items[ln]})}
        manifest["collisions"] += len(coll_items[ln])
    manifest["collision_mesh_files"] = mesh_count
    if cfg["fix_to_world"]:
        L.append('    <joint name="world_fixed" type="fixed">')
        L.append("      <parent>world</parent>")
        L.append(f"      <child>{root}</child>")
        L.append("    </joint>")
    for jn, j in joints.items():
        jc = cfg["joints"].get(jn, {})
        if j["mode"] == "fixed":
            L.append(f'    <joint name="{jn}" type="fixed">')
            L.append(f"      <parent>{j['parent']}</parent><child>{j['child']}</child>")
            L.append("    </joint>")
            manifest["joints"][jn] = {"type": j["type"], "mode": "fixed", "position": j["q"]}
            continue
        jtype = "revolute" if j["type"] in ("revolute", "continuous") else "prismatic"
        dflt = cfg["joint_defaults"].get(jtype, {})
        lower = jc.get("lower", j["lower"] if j["type"] != "continuous" else None)
        upper = jc.get("upper", j["upper"] if j["type"] != "continuous" else None)
        damping = float(jc.get("damping", dflt.get("damping", j["damping"] or 0.0)))
        friction = float(jc.get("friction", dflt.get("friction", j["friction"] or 0.0)))
        effort = float(jc.get("effort", -1))
        L.append(f'    <joint name="{jn}" type="{jtype}">')
        L.append(f"      <parent>{j['parent']}</parent>")
        L.append(f"      <child>{j['child']}</child>")
        L.append("      <axis>")
        L.append(f"        <xyz>{fmt(j['axis'])}</xyz>")
        if lower is not None and upper is not None:
            L.append(f"        <limit><lower>{lower:.6g}</lower><upper>{upper:.6g}</upper>"
                     f"<effort>{effort:.6g}</effort></limit>")
        else:
            L.append(f"        <limit><lower>-inf</lower><upper>inf</upper><effort>{effort:.6g}</effort></limit>")
        L.append(f"        <dynamics><damping>{damping:.6g}</damping><friction>{friction:.6g}</friction></dynamics>")
        L.append("      </axis>")
        L.append("    </joint>")
        axis_model = Tm[j["child"]][:3, :3] @ np.asarray(j["axis"])
        manifest["joints"][jn] = {
            "type": jtype, "mode": "live", "parent": j["parent"], "child": j["child"],
            "lower": lower, "upper": upper, "damping": damping, "friction": friction,
            "origin_model_frame": list(np.round(Tm[j["child"]][:3, 3], 5)),
            "axis_model_frame": list(np.round(axis_model, 6))}
        log(f"joint {jn}: {jtype} live, limits [{lower}, {upper}], damping {damping}, friction {friction}")
    for jn, j in joints.items():
        if j["mode"] == "fixed" and j["type"] != "fixed":
            log(f"joint {jn}: {j['type']} locked (fixed) at {j['q']}")
    L.append("  </model>")
    L.append("</sdf>")
    with open(os.path.join(out, "model.sdf"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "model.config"), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.11">model.sdf</sdf>
  <description>Generated by asset_pipeline/urdf_to_gz.py from {os.path.basename(inp)}. {cfg['description']}</description>
</model>
""")
    manifest["log"] = log_lines
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"wrote {out}/model.sdf: {manifest['collisions']} collisions "
        f"({mesh_count} mesh files), aabb {np.round(aabb, 3).tolist() if aabb else None}")
    return out, manifest


def geometry_sdf(g):
    t = g["type"]
    if t == "box":
        return f"<box><size>{fmt(g['size'])}</size></box>"
    if t == "cylinder":
        return f"<cylinder><radius>{g['radius']:.6g}</radius><length>{g['length']:.6g}</length></cylinder>"
    if t == "sphere":
        return f"<sphere><radius>{g['radius']:.6g}</radius></sphere>"
    if t == "mesh":
        return f"<mesh><uri>{g['uri']}</uri><scale>{fmt(g.get('scale', [1, 1, 1]))}</scale></mesh>"
    raise ValueError(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help=".urdf or Habitat *.object_config.json")
    ap.add_argument("out_dir", help="models are written to OUT_DIR/<name>/")
    ap.add_argument("--config", help="per-object YAML/JSON config (see configs/)")
    ap.add_argument("--name", help="model name (default: config name, else URDF robot name)")
    a = ap.parse_args()
    cfg = copy.deepcopy(DEFAULTS)
    if a.config:
        with open(a.config) as f:
            user = yaml.safe_load(f) or {}
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            sys.exit(f"unknown config keys {sorted(unknown)}")
        deep_update(cfg, user)
        cfg["_config_path"] = os.path.abspath(a.config)
    convert(a.input, a.out_dir, cfg, a.name)


if __name__ == "__main__":
    main()
