"""The gz model of the mobile SO-101: the SO-101 arm on the shared virtual holonomic base.

Kinematics, masses, limits and visual meshes are read from TheRobotStudio's so101_new_calib.urdf
(robot.yaml `description`, ThirdParty/so_arm100, Apache-2.0); the URDF's base_link is renamed
so101_base_link. Collisions are primitive boxes only (the URDF collides with the visual meshes): the ones
MuJoCo Menagerie's robotstudio_so101/so101.xml (Apache-2.0) puts on the same link frames, with the jaw
pads thickened outwards (inner faces kept). Every collision is in category 0x2 colliding with 0x1 only
(no self-collision), each link has a contact sensor. Link inertias get robot.INERTIA_ADD (a rotor inertia
stand-in: the gz MuJoCo plugin drops <armature>); the before/after values are printed and written to
inertias.json; robot.yaml arm.armature / gripper.armature replace them. robot.yaml `scale` s scales all of it (the URDF on load, meshes, boxes, the TCP; robot.py).
"""
import json
import math
import pathlib
import shutil

import numpy as np

from robots.base import base_sdf, tf
from robots.mobile_so101.robot import ARM, ARM_LINKS, INERTIA_ADD, JAW, JAW_DAMPING, JAW_LINK, RENAME, load_urdf

ARM_MASK = "<surface><contact><category_bitmask>2</category_bitmask>" \
           "<collide_bitmask>1</collide_bitmask></contact></surface>"
COLOURS = {"3d_printed": ("1 0.82 0.12 1", "1 0.82 0.12 1"), "sts3215": ("0.1 0.1 0.1 1", "0.1 0.1 0.1 1")}
VEL = 4.7                            # rad/s, STS3215 no-load speed (~0.22 s / 60 deg at 12 V)

# Menagerie boxes per link: (pos, quat wxyz, half sizes); the pads (names) are thickened to 5 mm
# outwards from their inner face (gripper_link -x, moving jaw +x).
COLLISIONS = {
    "shoulder_link": [((-0.030399, 0.000422, -0.0387), (1, 1, 1, -1), (0.023, 0.015, 0.01)),
                      ((-0.025, 0, 0.0), (1, 1, -1, 1), (0.038, 0.025, 0.02))],
    "upper_arm_link": [((-0.06, 0, 0.02), (0, -1, 1, 0), (0.01, 0.07, 0.03)),
                       ((-0.12, -0.014, 0.0182), (0, 1, 0, 0), (0.01, 0.02, 0.015))],
    "lower_arm_link": [((-0.05, 0, 0.0182), (0, 1, 0, 0), (0.07, 0.01, 0.03)),
                       ((-0.125, 0.005, 0.018), (0, -1, 0, 0), (0.023, 0.013, 0.018))],
    "wrist_link": [((0, -0.0424, 0.0256), (1, 1, 1, -1), (0.029, 0.015, 0.018)),
                   ((0, -0.02, 0.0191), (1, -1, -1, -1), (0.027, 0.01, 0.03))],
    "gripper_link": [((0, -0.000218, 0.00095), (0, 1, 0, 0), (0.01, 0.01, 0.018)),
                     ((-0.0025, 0, -0.022), (1, 0, 0, 0), (0.0325, 0.015, 0.015)),
                     ((-0.024, 0, -0.04), (1, 0, 0, 0), (0.01, 0.015, 0.005)),
                     ("pad", -0.008, -0.0982, (0.004, 0.004)),      # fixed_jaw_box4 (the tip pad)
                     ("pad", -0.0098, -0.0905, (0.005, 0.006)),     # box5
                     ("pad", -0.0115, -0.0727, (0.009, 0.008)),     # box6
                     ("pad", -0.0133, -0.053, (0.01, 0.008))],      # box7
    JAW_LINK: [((0, -0.013, 0.019), (1, 0, 0, 0), (0.01, 0.01, 0.015)),
               ("jaw_pad", -0.0123, -0.076, (0.004, 0.004)),     # moving_jaw_box2 (the tip pad)
               ("jaw_pad", -0.0103, -0.067, (0.005, 0.006))],    # box3
}
PAD_T = 0.005


def f(*v):
    return " ".join(f"{x:.6g}" for x in v)


def quat_rpy(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return mat_rpy(R)


def mat_rpy(R):
    p = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])
    return math.atan2(-R[1, 2], R[1, 1]), p, 0.0


def boxes(link, s=1.0):
    """[(name, pose text, size)] of a link's collision boxes, scaled by s."""
    out = []
    t = PAD_T * s
    for k, c in enumerate(COLLISIONS.get(link, [])):
        if c[0] == "pad":            # fixed jaw pad: inner face at x, 5 mm thick towards -x
            _, x, z, (hy, hz) = c
            x, z, hy, hz = x * s, z * s, hy * s, hz * s
            out.append((f"{link}_pad{k}", f(x - t / 2, 0, z, 0, 0, 0), (t, 2 * hy, 2 * hz)))
        elif c[0] == "jaw_pad":      # moving jaw pad (jaw frame): inner face at x, thick towards +x
            _, x, y, (hy, hz) = c
            x, y, hy, hz = x * s, y * s, hy * s, hz * s
            out.append((f"{link}_pad{k}", f(x + t / 2, y, 0.01875 * s, 0, 0, 0), (t, 2 * hy, 2 * hz)))
        else:
            pos, q, h = c
            out.append((f"{link}_box{k}", f(*(v * s for v in pos), *quat_rpy(*q)), tuple(2 * v * s for v in h)))
    return out


def main(robot, description, out_dir):
    URDF, OUT = pathlib.Path(description), pathlib.Path(out_dir)
    NAME = robot.model
    s = robot.scale
    U = load_urdf(URDF, s)
    report = {}
    armature = {k: v * s ** 5 for k, v in INERTIA_ADD.items()}
    if robot.spec.arm.armature is not None:
        armature.update(zip(ARM_LINKS, robot.spec.arm.armature))
    if robot.spec.gripper.armature is not None:
        armature[JAW_LINK] = robot.spec.gripper.armature

    def inertial(link):
        d = U["links"][link]
        i = dict(d["inertia"])
        add = armature.get(link, 0.0)
        before = {k: i[k] for k in ("ixx", "iyy", "izz")}
        I = np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]])
        e = np.sort(np.linalg.eigvalsh(I))
        pad = max(0.0, e[2] - e[0] - e[1]) * 1.05      # triangle inequality (as the Panda's model.inertial)
        for k in ("ixx", "iyy", "izz"):
            i[k] += pad + add
        report[link] = dict(mass=d["mass"], before=before, after={k: i[k] for k in ("ixx", "iyy", "izz")}, added=add)
        return f"""
      <inertial>
        <pose>{f(*d['com'], *d['com_rpy'])}</pose>
        <mass>{d['mass']}</mass>
        <inertia><ixx>{i['ixx']:.9g}</ixx><ixy>{i['ixy']:.9g}</ixy><ixz>{i['ixz']:.9g}</ixz>
          <iyy>{i['iyy']:.9g}</iyy><iyz>{i['iyz']:.9g}</iyz><izz>{i['izz']:.9g}</izz></inertia>
      </inertial>"""

    def visuals(link):
        out = []
        for k, v in enumerate(U["links"][link]["visuals"]):
            amb, dif = COLOURS.get(v["material"], COLOURS["3d_printed"])
            out.append(f"""
      <visual name="{link}_visual{k}">
        <pose>{f(*v['xyz'], *v['rpy'])}</pose>
        <geometry><mesh><uri>model://{NAME}/meshes/{pathlib.Path(v['mesh']).name}</uri>{mesh_scale(v)}</mesh></geometry>
        <material><ambient>{amb}</ambient><diffuse>{dif}</diffuse></material>
      </visual>""")
        return "".join(out)

    def mesh_scale(v):
        return f"<scale>{f(*[v['mesh_scale']] * 3)}</scale>" if "mesh_scale" in v else ""

    def collisions(link):
        cols = boxes(link, s)
        out = [f"""
      <collision name="{n}">
        <pose>{pose}</pose>
        <geometry><box><size>{f(*size)}</size></box></geometry>
        {ARM_MASK}
      </collision>""" for n, pose, size in cols]
        if cols:
            out.append(f"""
      <sensor name="contact" type="contact">
        <update_rate>200</update_rate>
        <contact>{''.join(f'<collision>{n}</collision>' for n, _, _ in cols)}</contact>
      </sensor>""")
        return "".join(out)

    def link(name, pose_rel):
        model_name = RENAME.get(name, name)
        return f"""
    <link name="{model_name}">
      <pose relative_to="{pose_rel}">0 0 0 0 0 0</pose>
      {inertial(name)}
      {visuals(name)}{collisions(name)}
    </link>"""

    parts = [base_sdf(robot, f"{inertial('base_link')}\n      {visuals('base_link')}")]
    jaw_damping = JAW_DAMPING if robot.spec.gripper.damping is None else robot.spec.gripper.damping
    damping = dict(zip(ARM, robot.arm_damping), **{JAW: jaw_damping})
    for jn in (*ARM, JAW):
        j = U["joints"][jn]
        parent, child = RENAME.get(j["parent"], j["parent"]), j["child"]
        parts.append(f"""
    <joint name="{jn}" type="revolute">
      <pose relative_to="{parent}">{f(*j['xyz'], *j['rpy'])}</pose>
      <parent>{parent}</parent>
      <child>{child}</child>
      <axis>
        <xyz>{f(*j['axis'])}</xyz>
        <limit><lower>{j['lower']}</lower><upper>{j['upper']}</upper>
          <effort>-1</effort><velocity>{VEL}</velocity></limit>
        <dynamics><damping>{damping[jn]}</damping></dynamics>
      </axis>
    </joint>{link(child, jn)}""")

    # The TCP (so101_tcp) and the hand frame link the gz systems read it from (TCP = hand @ (0, 0, tcp_z))
    TCP_IN_GRIPPER = robot.tcp_in_gripper
    T_hand = TCP_IN_GRIPPER @ tf(p=(0, 0, -robot.tcp_z))
    parts.append(f"""
    <frame name="{robot.tcp_link}" attached_to="gripper_link">
      <pose>{f(*TCP_IN_GRIPPER[:3, 3], *mat_rpy(TCP_IN_GRIPPER[:3, :3]))}</pose>
    </frame>
    <joint name="so101_hand_joint" type="fixed">
      <pose relative_to="gripper_link">{f(*T_hand[:3, 3], *mat_rpy(T_hand[:3, :3]))}</pose>
      <parent>gripper_link</parent>
      <child>{robot.hand_link}</child>
    </joint>
    <link name="{robot.hand_link}">
      <pose relative_to="so101_hand_joint">0 0 0 0 0 0</pose>
      <inertial><mass>0.001</mass>
        <inertia><ixx>1e-7</ixx><iyy>1e-7</iyy><izz>1e-7</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
    </link>
    <plugin filename="gz-sim-joint-state-publisher-system"
            name="gz::sim::systems::JointStatePublisher"/>""")

    sdf = f"""<?xml version="1.0"?>
<!-- Generated by gen_so101_sdf.py from TheRobotStudio SO-ARM100 so101_new_calib.urdf (Apache-2.0);
     collision boxes after MuJoCo Menagerie robotstudio_so101 (Apache-2.0). -->
<sdf version="1.11">
  <model name="{NAME}">
{''.join(parts)}
  </model>
</sdf>
"""
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    for ln in U["links"].values():
        for v in ln["visuals"]:
            shutil.copy(URDF.parent / v["mesh"], OUT / "meshes" / pathlib.Path(v["mesh"]).name)
    (OUT / "model.sdf").write_text(sdf)
    (OUT / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{NAME}</name>
  <version>1.0</version>
  <sdf version="1.11">model.sdf</sdf>
  <description>SO-101 arm on a virtual holonomic base, generated from so101_new_calib.urdf.</description>
</model>
""")
    (OUT / "inertias.json").write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        print(f"  {k}: m {v['mass']:.3f} kg, I diag {[f'{x:.3g}' for x in v['before'].values()]} -> "
              f"{[f'{x:.3g}' for x in v['after'].values()]} kg m^2")
    print(f"Wrote {OUT / 'model.sdf'}")
