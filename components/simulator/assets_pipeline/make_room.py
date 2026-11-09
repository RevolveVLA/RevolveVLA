#!/usr/bin/env python3
"""Kitchen room shell (visual upgrade) for the kitchen worlds, one model per asset set:
assets/generated/kitchen_room_<set>/ (git-ignored). The worlds include it under the
name `walls`. Its link carries the three wall collisions of the old inline `walls`
model unchanged; everything else is <visual> or <light> (the MuJoCo plugin builds
geoms from <collision> only, so physics is unchanged).

Textures: CC0 PBR sets from ambientCG (pinned IDs, URLs and zip sha256 below),
downloaded once into assets/textures/<id>/ (git-ignored). Offline or on a checksum
mismatch the model uses plain colours and one warning line is printed. Idempotent:
existing textures are reused; files are only rewritten when their content changes.
Stdlib only (runs on the host). Downloads are untrusted data: only the
three named JPG maps are read out of each zip, nothing is executed.
  python3 -I components/simulator/assets_pipeline/make_room.py [--offline]
"""
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))   # the repo root
TEX_DIR = os.path.join(ROOT, "assets", "textures")
GEN_DIR = os.path.join(ROOT, "assets", "generated")
TIMEOUT = 30  # s per download

# key: (ambientCG asset id, zip sha256, fallback rgb). Licence: CC0 1.0 (ambientcg.com/license)
TEXTURES = {
    "floor": ("WoodFloor051", "3f493484eab1ec5e1c466b90e515003b286fc6d7f84ff8ff6485900bfc26cef5", "0.62 0.45 0.30"),
    "wall": ("Plaster001", "944b4831016e42ace4a89422e4e7190912ca2f7fed6f561354c48ec7bb54d3a4", "0.80 0.78 0.74"),
    "splash": ("Tiles010", "14d87fcd62d10977b5e57778d5271912e0fc3eab702d4ffbc55855e8a84c6280", "0.92 0.92 0.90"),
    "rug": ("Carpet012", "f5c459f278a3b50dc5c2bf535b8c3b6fd4bb6fc458c836b64023961b167fe4f6", "0.25 0.28 0.33"),
}
URL = "https://ambientcg.com/get?file={id}_1K-JPG.zip"
MAPS = ("Color", "NormalGL", "Roughness")

# Collisions of the old inline `walls` model (worlds/kitchen*.sdf up to 7dd901f), verbatim.
WALL_COLLISIONS = {
    "west": """        <collision name="west"><pose>-3.05 0 1.1 0 0 0</pose>
          <geometry><box><size>0.1 6.2 2.2</size></box></geometry></collision>""",
    "north": """        <collision name="north"><pose>0 3.05 1.1 0 0 0</pose>
          <geometry><box><size>6.0 0.1 2.2</size></box></geometry></collision>""",
    "east": """        <collision name="east"><pose>3.05 0 1.1 0 0 0</pose>
          <geometry><box><size>0.1 6.2 2.2</size></box></geometry></collision>""",
}
WALL_BOXES = {"west": ("-3.05 0 1.1 0 0 0", "0.1 6.2 2.2"), "north": ("0 3.05 1.1 0 0 0", "6.0 0.1 2.2"),
              "east": ("3.05 0 1.1 0 0 0", "0.1 6.2 2.2")}

H = 2.8  # visual wall / ceiling height (collisions stay 2.2 m); cameras sit at <= 2.6 m

# Per asset set: worktop height and extent along the west wall, upper-cabinet units,
# ceiling spot positions. Decor stays against the walls (x < -2.66 west, on the north
# wall face) or flat on the floor (rug), clear of the robot's paths and the cameras.
SETS = {
    "partnet": {"top": 0.900, "y": (-0.10, 0.68), "cabs": [(-0.10, 0.68)],
                "spots": [(-2.2, 0.3), (-1.5, 1.8), (1.4, -0.5), (0.2, 0.3)]},
}


def warn_once(msg, _done=[]):
    if not _done:
        print(f"WARNING: kitchen textures: {msg}; using plain colours", file=sys.stderr)
        _done.append(1)


def fetch(asset_id, sha, offline):
    """Return {map: path} for the asset, downloading it if needed; None if unavailable."""
    d = os.path.join(TEX_DIR, asset_id)
    paths = {m: os.path.join(d, f"{asset_id}_{m}.jpg") for m in MAPS}
    if all(os.path.isfile(p) for p in paths.values()):
        return paths
    if offline:
        warn_once("missing and --offline")
        return None
    try:
        req = urllib.request.Request(URL.format(id=asset_id), headers={"User-Agent": "kitchen-make-room/1"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read()
    except Exception as e:  # network down, HTTP error, timeout
        warn_once(f"download of {asset_id} failed ({e.__class__.__name__}: {e})")
        return None
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        warn_once(f"{asset_id} zip sha256 {got[:12]} != pinned {sha[:12]}")
        return None
    os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        for m, p in paths.items():
            src = names.get(f"{asset_id}_1K-JPG_{m}.jpg")
            if src is None:
                warn_once(f"{asset_id} zip has no {m} map")
                return None
            tmp = f"{p}.{os.getpid()}.part"  # parallel runs: unique temp, atomic rename
            with open(tmp, "wb") as f:
                f.write(z.read(src))
            os.replace(tmp, p)
    return paths


def write(path, data):
    """Write only when the content changes (keeps mtimes stable across runs)."""
    mode = "b" if isinstance(data, bytes) else ""
    if os.path.isfile(path):
        with open(path, "r" + mode) as f:
            if f.read() == data:
                return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "w" + mode) as f:
        f.write(data)
    os.replace(tmp, path)


def quad(corners, tile):
    """OBJ quad; corners counter-clockwise seen from the lit side; UVs repeat every `tile` m."""
    sub = lambda a, b: [a[i] - b[i] for i in range(3)]
    norm = lambda v: sum(x * x for x in v) ** 0.5
    u, v = sub(corners[1], corners[0]), sub(corners[3], corners[0])
    n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    n = [x / norm(n) for x in n]
    w, h = norm(u) / tile, norm(v) / tile
    s = "".join("v %.4f %.4f %.4f\n" % tuple(c) for c in corners)
    s += "vt 0 0\nvt %.4f 0\nvt %.4f %.4f\nvt 0 %.4f\n" % (w, w, h, h)
    s += "vn %.4f %.4f %.4f\n" % tuple(n)
    return s + "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n"


def material(key, tex, name, tint="1 1 1", rough=None):
    rgb = TEXTURES[key][2]
    if tex.get(key) is None:
        r = 0.6 if rough is None else rough
        return (f"<material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse><specular>0.1 0.1 0.1 1</specular>"
                f"<pbr><metal><roughness>{r}</roughness><metalness>0</metalness></metal></pbr></material>")
    uri = lambda m: f"model://{name}/materials/textures/{TEXTURES[key][0]}_{m}.jpg"
    r = f"<roughness>{rough}</roughness>" if rough is not None else f"<roughness_map>{uri('Roughness')}</roughness_map>"
    return (f"<material><ambient>{tint} 1</ambient><diffuse>{tint} 1</diffuse><specular>0.2 0.2 0.2 1</specular>"
            f"<pbr><metal><albedo_map>{uri('Color')}</albedo_map><normal_map>{uri('NormalGL')}</normal_map>"
            f"{r}<metalness>0</metalness></metal></pbr></material>")


def plain(rgb, rough=0.5, metal=0.0, emissive=None):
    em = f"<emissive>{emissive} 1</emissive>" if emissive else ""
    return (f"<material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse><specular>0.3 0.3 0.3 1</specular>{em}"
            f"<pbr><metal><roughness>{rough}</roughness><metalness>{metal}</metalness></metal></pbr></material>")


def mesh_visual(vname, name, mat, shadows=True):
    cs = "" if shadows else "<cast_shadows>false</cast_shadows>"
    return (f'        <visual name="{vname}">{cs}\n          <geometry><mesh><uri>model://{name}/meshes/{vname}.obj</uri>'
            f"</mesh></geometry>\n          {mat}</visual>\n")


def box_visual(vname, pose, size, mat, shadows=True):
    cs = "" if shadows else "<cast_shadows>false</cast_shadows>"
    return (f'        <visual name="{vname}">{cs}<pose>{pose}</pose>\n          <geometry><box><size>{size}</size></box>'
            f"</geometry>\n          {mat}</visual>\n")


def build(set_name, tex):
    cfg = SETS[set_name]
    name = f"kitchen_room_{set_name}"
    out = os.path.join(GEN_DIR, name)
    meshes, vis = {}, ""
    # inner faces 2 mm in front of the wall boxes, floor 1 mm above z = 0, ceiling at H
    meshes["floor"] = quad([(-3, -3.1, 0.001), (3, -3.1, 0.001), (3, 3, 0.001), (-3, 3, 0.001)], 1.6)
    meshes["wall_w"] = quad([(-2.998, -3.1, 0), (-2.998, 3, 0), (-2.998, 3, H), (-2.998, -3.1, H)], 1.5)
    meshes["wall_n"] = quad([(-3, 2.998, 0), (3, 2.998, 0), (3, 2.998, H), (-3, 2.998, H)], 1.5)
    meshes["wall_e"] = quad([(2.998, 3, 0), (2.998, -3.1, 0), (2.998, -3.1, H), (2.998, 3, H)], 1.5)
    meshes["ceiling"] = quad([(-3, 3, H), (3, 3, H), (3, -3.1, H), (-3, -3.1, H)], 2.0)
    y0, y1 = cfg["y"]
    meshes["backsplash"] = quad([(-2.996, y0, cfg["top"]), (-2.996, y1, cfg["top"]), (-2.996, y1, 1.50),
                                 (-2.996, y0, 1.50)], 0.6)
    meshes["rug"] = quad([(0.5, -1.2, 0.003), (2.5, -1.2, 0.003), (2.5, 0.2, 0.003), (0.5, 0.2, 0.003)], 0.7)

    for k, (pose, size) in WALL_BOXES.items():  # the old wall boxes (outside and top faces)
        vis += box_visual(k, pose, size, plain("0.86 0.84 0.79", 0.8))
    vis += mesh_visual("floor", name, material("floor", tex, name))
    for k in ("wall_w", "wall_n", "wall_e"):
        vis += mesh_visual(k, name, material("wall", tex, name, tint="1.0 0.97 0.92"))
    vis += mesh_visual("ceiling", name, plain("0.95 0.94 0.92", 0.9), shadows=False)
    vis += mesh_visual("backsplash", name, material("splash", tex, name))
    vis += mesh_visual("rug", name, material("rug", tex, name, tint="1.0 0.92 0.85"), shadows=False)
    # baseboards
    trim = plain("0.94 0.93 0.90", 0.4)
    vis += box_visual("base_w", "-2.99 -0.05 0.05 0 0 0", "0.02 6.1 0.10", trim)
    vis += box_visual("base_n", "0 2.99 0.05 0 0 0", "5.96 0.02 0.10", trim)
    vis += box_visual("base_e", "2.99 -0.05 0.05 0 0 0", "0.02 6.1 0.10", trim)
    # upper wall cabinets above the worktop (x -3.0 .. -2.66, z 1.55 .. 2.25): carcass, door(s), bar handles
    cab, door, steel = plain("0.30 0.37 0.33", 0.5), plain("0.34 0.42 0.37", 0.4), plain("0.75 0.75 0.75", 0.25, 1.0)
    for i, (a, b) in enumerate(cfg["cabs"]):
        c, w = (a + b) / 2, b - a
        vis += box_visual(f"cab{i}", f"-2.83 {c:.3f} 1.90 0 0 0", f"0.34 {w:.3f} 0.70", cab)
        ndoor = 2 if w > 0.6 else 1
        dw = (w - 0.01 * (ndoor + 1)) / ndoor
        for j in range(ndoor):
            yc = a + 0.01 + dw / 2 + j * (dw + 0.01)
            vis += box_visual(f"cab{i}_door{j}", f"-2.655 {yc:.3f} 1.90 0 0 0", f"0.012 {dw:.3f} 0.68", door)
            hy = yc + (dw / 2 - 0.05) * (-1 if ndoor == 2 and j == 1 else 1)  # at the meeting edge
            vis += box_visual(f"cab{i}_handle{j}", f"-2.64 {hy:.3f} 1.66 0 0 0", "0.015 0.012 0.14", steel)
    # window on the north wall (east of the fridge): frame, emissive daylight pane, mullion, sill
    white = plain("0.95 0.95 0.95", 0.4)
    vis += box_visual("win_frame", "0.9 2.985 1.45 0 0 0", "1.30 0.03 1.10", white)
    vis += box_visual("win_pane", "0.9 2.968 1.45 0 0 0", "1.16 0.01 0.96",
                      plain("0.75 0.85 0.95", 0.05, 0, "0.70 0.78 0.86"), shadows=False)
    vis += box_visual("win_mullion", "0.9 2.962 1.45 0 0 0", "0.04 0.01 0.96", white)
    vis += box_visual("win_sill", "0.9 2.95 0.88 0 0 0", "1.40 0.10 0.03", white)
    # ceiling lamps: emissive discs at the spot lights
    lamp = plain("1 0.95 0.85", 0.3, 0, "1.0 0.92 0.78")
    lights = ""
    for i, (x, y) in enumerate(cfg["spots"]):
        vis += (f'        <visual name="lamp{i}"><cast_shadows>false</cast_shadows><pose>{x} {y} {H - 0.01} 0 0 0</pose>\n'
                f"          <geometry><cylinder><radius>0.09</radius><length>0.02</length></cylinder></geometry>\n"
                f"          {lamp}</visual>\n")
        lights += (f'      <light type="spot" name="fill{i}">\n'
                   f"        <cast_shadows>false</cast_shadows><pose>{x} {y} {H - 0.05} 0 0 0</pose>\n"
                   f"        <diffuse>1.0 0.90 0.76 1</diffuse><specular>0.25 0.22 0.2 1</specular>\n"
                   f"        <intensity>1.4</intensity><direction>0 0 -1</direction>\n"
                   f"        <attenuation><range>8</range><constant>0.5</constant><linear>0.08</linear>"
                   f"<quadratic>0.02</quadratic></attenuation>\n"
                   f"        <spot><inner_angle>0.7</inner_angle><outer_angle>1.4</outer_angle><falloff>1.0</falloff></spot>\n"
                   f"      </light>\n")

    sdf = f"""<?xml version="1.0"?>
<!-- Generated by asset_pipeline/make_room.py ({set_name}); do not edit. Included by
     worlds/kitchen*.sdf under the name `walls`. Collisions = the old inline walls. -->
<sdf version="1.11">
  <model name="{name}">
    <static>true</static>
    <link name="link">
{chr(10).join(WALL_COLLISIONS.values())}
{vis}{lights}    </link>
  </model>
</sdf>
"""
    write(os.path.join(out, "model.sdf"), sdf)
    write(os.path.join(out, "model.config"),
          f'<?xml version="1.0"?>\n<model><name>{name}</name><version>1.0</version><sdf version="1.11">model.sdf</sdf>\n'
          "<description>Kitchen room shell (visual only beyond the wall collisions); "
          "textures CC0 from ambientCG</description></model>\n")
    for k, s in meshes.items():
        write(os.path.join(out, "meshes", f"{k}.obj"), s)
    for key, paths in tex.items():
        if paths:
            for p in paths.values():
                with open(p, "rb") as f:
                    write(os.path.join(out, "materials", "textures", os.path.basename(p)), f.read())


def main():
    offline = "--offline" in sys.argv[1:]
    tex = {}
    for k, (aid, sha, _) in TEXTURES.items():
        tex[k] = fetch(aid, sha, offline)
        if tex[k] is None:  # all or nothing (one consistent look); stop after the first failure
            tex = {k: None for k in TEXTURES}
            break
    for s in SETS:
        build(s, tex)


if __name__ == "__main__":
    main()
