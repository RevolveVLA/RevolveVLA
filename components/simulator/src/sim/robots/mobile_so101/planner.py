"""cuRobo planner robot of the mobile SO-101 (datagen: gen_assets.robot() adds the shared base chain, the base
spheres and the kinematics cfg; robots/base.py Robot.planner loads this module). Runs in revolve-datagen only.

  chain(robot, rd)     so101_base_link -> the 5 arm joints and the jaw (so101_new_calib.urdf origins and axes,
                       the gz model's limits) -> so101_tcp on gripper_link (TCP_IN_GRIPPER), all x robot.yaml scale
  self_collision(robot, rd, links)   links up to two joints apart along the chain ignore each other
  BOX_CENTRE_OBJ       start design: none (generator.yaml drops the starts block for this robot: a start IK
                       with the base locked has no solution for a 5-DoF arm off its plane)

Collision spheres: cuRobo's own fit (Cuboid.get_bounding_spheres) of the gz model's collision boxes
(model.py COLLISIONS, read from the generated model.sdf, robot.yaml gz_sdf), except the fixed jaw's pads: the
TCP sits 8 mm above the fixed pad face, so their spheres plus the sphere buffer would overlap the 14 mm bar of
every grasp (the jaw tips are the ones meant to touch it). The fit samples box surfaces with trimesh's
sample_surface_even, which cuRobo calls without a seed (OS entropy): FIT_SEED is passed to it while fitting so the
planner assets are reproducible.
"""
import functools
import math
from unittest import mock

import gen_assets as ga
import specs_bridge as plugins
from robots.mobile_so101.robot import ARM, ARM_LINKS, JAW, JAW_LINK, RENAME, load_urdf

BOX_CENTRE_OBJ = {}
SKIP_BOXES = ("gripper_link_pad",)     # fixed jaw pads (see the module doc)
SPHERE_PITCH = 0.02                    # one fitted sphere per this much (x scale) of a box's longest side [m]
MAX_SPHERES_PER_BOX = 4
FIT_SEED = 0                           # seed of every box's surface samples (see the module doc)


def spheres_of(box_list, pitch=SPHERE_PITCH):
    """cuRobo-fitted spheres [{center, radius}] (link frame) of [(name, T in link, size)] boxes."""
    import trimesh
    from curobo.geom.types import Cuboid
    out = []
    even = functools.partial(trimesh.sample.sample_surface_even, seed=FIT_SEED)
    for name, T, size in box_list:
        if name.startswith(SKIP_BOXES):
            continue
        n = int(min(MAX_SPHERES_PER_BOX, max(1, math.ceil(max(size) / pitch))))
        with mock.patch.object(trimesh.sample, "sample_surface_even", even):
            fit = Cuboid(name=name, pose=ga.pose7(T), dims=[float(v) for v in size]).get_bounding_spheres(n_spheres=n)
        for s in fit:
            out.append({"center": [float(v) for v in s.pose[:3]], "radius": float(s.radius)})
    return out


def chain(robot, rd):
    """dict(links: link names root..TCP, joints: URDF joints, spheres: {link: [{center, radius}]})."""
    U = load_urdf(plugins.path(rd["description"]), robot.scale)
    gz = ga.sdf.Model(plugins.path(rd["gz_sdf"]))           # the gz model (entry.py generates it first)
    joints = []
    for n in (*ARM, JAW):
        u, g = U["joints"][n], gz.joints[n]
        joints.append(ga.joint(n, "revolute", RENAME.get(u["parent"], u["parent"]), u["child"],
                               ga.sdf.tf(u["xyz"], u["rpy"]), u["axis"], (g["lower"], g["upper"], g["velocity"])))
    joints.append(ga.joint("so101_tcp_joint", "fixed", "gripper_link", rd["tcp"]["link"], robot.tcp_in_gripper))
    arm = [*ARM_LINKS, JAW_LINK]
    spheres = {ln: spheres_of(gz.boxes.get(ln, []), SPHERE_PITCH * robot.scale) for ln in arm}
    return dict(links=[robot.root_link, *arm, rd["tcp"]["link"]], joints=joints,
                spheres={ln: s for ln, s in spheres.items() if s})


def self_collision(robot, rd, links):
    """(ignore, buffer) over base_link + the arm's sphere links: each ignores the next two along the chain."""
    order = ["base_link", robot.root_link, *ARM_LINKS, JAW_LINK]
    has = {"base_link", *links}
    ignore = {ln: [o for o in order[i + 1:i + 3] if o in has] for i, ln in enumerate(order) if ln in has}
    return ignore, {ln: 0.0 for ln in order if ln in has}
