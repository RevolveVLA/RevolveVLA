"""The gz model of the mobile Panda: Franka Panda (FER) + Franka Hand on the shared virtual base.

Kinematics, inertials and limits are read from franka_description (Apache-2.0, robot.yaml
`description`). Visuals use the original .dae meshes. Collisions use primitives only: the capsules
franka_description defines for self-collision, and its finger boxes. Primitive collisions map 1:1 onto
native MuJoCo geoms, and avoid gz-physics issue #1162 (shared mesh files crash the MuJoCo compile).
Every collision link has a contact sensor. Arm effort limits are left unset: with the pinned plugin, a
servo saturating at an <effort> limit (MuJoCo actfrcrange) falls into a ~1 Hz limit cycle under gravity.

main(robot, description, out_dir, mobile=False) writes the fixed-base model "panda" of the Panda-only
can-on-plate demo (can_demo/): the arm on the world, no contact sensors.
"""
import math
import os
import pathlib
import shutil

import numpy as np
import yaml

from robots.base import base_sdf

# Self-collision capsules from robots/common/franka_arm.xacro:
# (link, center xyz, axis, radius, cylinder length)
CAPSULES = {
    "link1": [((0, 0, -0.1915), "z", 0.06, 0.283)],
    "link2": [((0, 0, 0), "z", 0.06, 0.12)],
    "link3": [((0, 0, -0.145), "z", 0.06, 0.15)],
    "link4": [((0, 0, 0), "z", 0.06, 0.12)],
    "link5": [((0, 0, -0.26), "z", 0.06, 0.10),
              ((0, 0.08, -0.13), "z", 0.025, 0.14)],
    "link6": [((0, 0, -0.03), "z", 0.05, 0.08)],
    "link7": [((0, 0, 0.01), "z", 0.04, 0.14),
              ((0.06, 0, 0.082), "x", 0.03, 0.01)],
}

# Finger collision boxes from end_effectors/common/franka_hand.xacro:
# (xyz, rpy, size). The right finger mirrors the angled box.
FINGER_BOXES = [
    ((0, 0.0185, 0.011), (0, 0, 0), (0.022, 0.015, 0.020)),
    ((0, 0.0068, 0.0022), (0, 0, 0), (0.022, 0.0088, 0.0038)),
    ((0, 0.0159, 0.02835), (math.pi / 6, 0, 0), (0.0175, 0.007, 0.0235)),
    ((0, 0.00758, 0.04525), (0, 0, 0), (0.0175, 0.0152, 0.0185)),
]

# Arm links collide with the world but not with each other: they are in
# category 0x2 and only collide with category 0x1 (everything else in the
# world uses the default all-ones mask).
ARM_MASK = "<surface><contact><category_bitmask>2</category_bitmask>" \
           "<collide_bitmask>1</collide_bitmask></contact></surface>"

AXIS_RPY = {"z": (0, 0, 0), "x": (0, math.pi / 2, 0), "y": (math.pi / 2, 0, 0)}


def f(*v):
    return " ".join(f"{x:.6g}" for x in v)


def inertial(d):
    i = dict(d["inertia"])
    # Some franka_description tensors (link4) violate the triangle inequality,
    # which sdformat rejects. Add the smallest isotropic term that fixes it.
    I = np.array([[i["xx"], i["xy"], i["xz"]], [i["xy"], i["yy"], i["yz"]],
                  [i["xz"], i["yz"], i["zz"]]])
    e = np.sort(np.linalg.eigvalsh(I))
    pad = max(0.0, e[2] - e[0] - e[1]) * 1.05
    for k in ("xx", "yy", "zz"):
        i[k] += pad
    return f"""
      <inertial>
        <pose>{d['origin']['xyz']} {d['origin']['rpy']}</pose>
        <mass>{d['mass']}</mass>
        <inertia><ixx>{i['xx']}</ixx><ixy>{i['xy']}</ixy><ixz>{i['xz']}</ixz>
          <iyy>{i['yy']}</iyy><iyz>{i['yz']}</iyz><izz>{i['zz']}</izz></inertia>
      </inertial>"""


def main(robot, description, out_dir, mobile=True):
    FD, OUT = pathlib.Path(description), pathlib.Path(out_dir)
    NAME = robot.model if mobile else "panda"
    kin = yaml.safe_load((FD / "robots/fer/kinematics.yaml").read_text())
    inert = yaml.safe_load((FD / "robots/fer/inertials.yaml").read_text())
    limits = yaml.safe_load((FD / "robots/fer/joint_limits.yaml").read_text())
    hand_inert = yaml.safe_load((FD / "end_effectors/franka_hand/inertials.yaml").read_text())
    ARM_DAMPING = robot.arm_damping

    def visual(name, mesh, pose="0 0 0 0 0 0"):
        return f"""
      <visual name="{name}_visual">
        <pose>{pose}</pose>
        <geometry><mesh><uri>model://{NAME}/meshes/{mesh}</uri></mesh></geometry>
      </visual>"""

    def contact_sensor(names):
        """Contact sensor on the listed collisions (mobile model only)."""
        if not mobile:
            return ""
        cols = "".join(f"<collision>{n}</collision>" for n in names)
        return f"""
      <sensor name="contact" type="contact">
        <update_rate>200</update_rate>
        <contact>{cols}</contact>
      </sensor>"""

    def capsules(link):
        out = []
        for k, (xyz, axis, r, length) in enumerate(CAPSULES.get(link, [])):
            out.append(f"""
      <collision name="{link}_capsule{k}">
        <pose>{f(*xyz, *AXIS_RPY[axis])}</pose>
        <geometry><capsule><radius>{r}</radius><length>{length}</length></capsule></geometry>
        {ARM_MASK}
      </collision>""")
        return "".join(out)

    def finger_link(name, yaw):
        cols = []
        for k, (xyz, rpy, size) in enumerate(FINGER_BOXES):
            if os.environ.get("PADS_ONLY") and k != 3:
                continue
            rpy = (-rpy[0], rpy[1], math.pi) if (name == "right" and k == 2) else rpy
            cols.append(f"""
      <collision name="{name}finger_box{k}">
        <pose>{f(*xyz, *rpy)}</pose>
        <geometry><box><size>{f(*size)}</size></box></geometry>
      </collision>""")
        names = [f"{name}finger_box{k}" for k in range(len(FINGER_BOXES))
                 if not (os.environ.get("PADS_ONLY") and k != 3)]
        fi = dict(hand_inert["leftfinger"])
        return f"""
    <link name="panda_{name}finger">
      <pose relative_to="panda_finger_joint_{name}">0 0 0 0 0 0</pose>
      {inertial(fi)}
      {visual(name + 'finger', 'hand/finger.dae')}
      {''.join(cols)}{contact_sensor(names)}
    </link>
    <joint name="panda_finger_joint_{name}" type="prismatic">
      <pose relative_to="panda_hand">0 0 0.0584 0 0 {yaw}</pose>
      <parent>panda_hand</parent>
      <child>panda_{name}finger</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>0</lower><upper>0.04</upper><effort>40</effort><velocity>0.2</velocity></limit>
        <dynamics><damping>2.0</damping></dynamics>
      </axis>
    </joint>"""

    parts = []
    if mobile:
        parts.append(base_sdf(robot, f"{inertial(inert['link0'])}\n      {visual('link0', 'arm/link0.dae')}"))
    else:
        parts.append(f"""
    <link name="panda_link0">
      {inertial(inert['link0'])}
      {visual('link0', 'arm/link0.dae')}
    </link>
    <joint name="panda_base_fixed" type="fixed">
      <parent>world</parent>
      <child>panda_link0</child>
    </joint>""")

    for n in range(1, 8):
        j = kin[f"joint{n}"]["kinematic"]
        lim = limits[f"joint{n}"]["limit"] if "limit" in limits[f"joint{n}"] \
            else limits[f"joint{n}"]
        parts.append(f"""
    <joint name="panda_joint{n}" type="revolute">
      <pose relative_to="panda_link{n-1}">{f(j['x'], j['y'], j['z'], j['roll'], j['pitch'], j['yaw'])}</pose>
      <parent>panda_link{n-1}</parent>
      <child>panda_link{n}</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit><lower>{lim['lower']}</lower><upper>{lim['upper']}</upper>
          <effort>-1</effort><velocity>{lim['velocity']}</velocity></limit>
        <dynamics><damping>{ARM_DAMPING[n - 1]}</damping></dynamics>
      </axis>
    </joint>
    <link name="panda_link{n}">
      <pose relative_to="panda_joint{n}">0 0 0 0 0 0</pose>
      {inertial(inert[f'link{n}'])}
      {visual(f'link{n}', f'arm/link{n}.dae')}
      {capsules(f'link{n}')}{contact_sensor(
          [f"link{n}_capsule{k}" for k in range(len(CAPSULES.get(f"link{n}", [])))])}
    </link>""")

    # Flange (joint8, z=0.107) and hand_joint (yaw -pi/4) folded into one fixed joint
    flange = kin["joint8"]["kinematic"]["z"]
    parts.append(f"""
    <joint name="panda_hand_joint" type="fixed">
      <pose relative_to="panda_link7">0 0 {flange} 0 0 {-math.pi/4:.9f}</pose>
      <parent>panda_link7</parent>
      <child>panda_hand</child>
    </joint>
    <link name="panda_hand">
      <pose relative_to="panda_hand_joint">0 0 0 0 0 0</pose>
      {inertial(hand_inert['hand'])}
      {visual('hand', 'hand/hand.dae')}
      <collision name="hand_body">
        <pose>0 0 0.03 0 0 0</pose>
        <geometry><box><size>0.06 0.20 0.055</size></box></geometry>
        {ARM_MASK}
      </collision>{contact_sensor(["hand_body"])}
    </link>
    <frame name="panda_tcp" attached_to="panda_hand">
      <pose>0 0 {robot.tcp_z} 0 0 0</pose>
    </frame>""")
    parts.append(finger_link("left", 0))
    parts.append(finger_link("right", math.pi))

    parts.append("""
    <plugin filename="gz-sim-joint-state-publisher-system"
            name="gz::sim::systems::JointStatePublisher"/>""")

    sdf = f"""<?xml version="1.0"?>
<!-- Generated by gen_panda_sdf.py from franka_description (Apache-2.0). -->
<sdf version="1.11">
  <model name="{NAME}">
{''.join(parts)}
  </model>
</sdf>
"""

    (OUT / "meshes/arm").mkdir(parents=True, exist_ok=True)
    (OUT / "meshes/hand").mkdir(parents=True, exist_ok=True)
    for m in (FD / "meshes/robots/fer/visual").glob("link*.dae"):
        shutil.copy(m, OUT / "meshes/arm" / m.name)
    for m in ("hand.dae", "finger.dae"):
        shutil.copy(FD / "meshes/robot_ee/franka_hand_white/visual" / m,
                    OUT / "meshes/hand" / m)
    (OUT / "model.sdf").write_text(sdf)
    DESC = ("Franka Panda (FER) + Franka Hand on a virtual holonomic base"
            if mobile else "Franka Panda (FER) + Franka Hand")
    (OUT / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{NAME}</name>
  <version>1.0</version>
  <sdf version="1.11">model.sdf</sdf>
  <description>{DESC}, generated from franka_description.</description>
</model>
""")
    print(f"Wrote {OUT/'model.sdf'}")
