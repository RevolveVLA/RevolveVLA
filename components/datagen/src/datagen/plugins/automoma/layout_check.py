"""Pre-plan check of a layout (start_design.parse_layout; the realised randomisers): the robot chassis at the
layout's base spot must keep the task's replay.spawn_clearance_min_m from every static scene box.

  reject = clearance_reject("open_fridge_door", "mobile_panda")   # loads the specs and the scene boxes once
  reject({"start_pose": (x, y, yaw)})                # object frame -> None, or "collision: ..." (the reason)

Footprint only: the chassis box (robot.yaml chassis.size_xy, 5 mm above the floor up to mount z) at the spot
(start_design.layout_spot, object frame -> world as frames.py), grown so that it covers every attempt of the seed (the starts
block's offset_m and offset_deg, start_design.layout_starts). The arm and the replay's spawn stay with
the planner and the replay's spawn check. Scene boxes as the replay's clearance (sim/replay.py): the
task object and the `detailed` models box by box (joints closed), other models as their manifest AABB,
no floor. The distance is a lower bound: the chassis surface sampled every SPACING_M, less half a grid
cell diagonal. numpy, stdlib and PyYAML only (the manager or the host runs it while sampling); needs
the scene assets (assets/, not in git).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

import frames
import specs_bridge
import start_design
from revolve_common import specs

SIM = Path(__file__).resolve().parents[5] / "simulator" / "src" / "sim"
if str(SIM) not in sys.path:
    sys.path.append(str(SIM))           # the simulator's numpy-only sdf
import sdf  # noqa: E402

SPACING_M = 0.005


def scene_boxes(world, model_dir, skip=(), detailed=(), obj=None):
    """[(name, 4x4 world pose, size)] of a gz world file's static boxes; model_dir(uri) -> the model's dir."""
    out = []
    for name, uri, T, inline in sdf.world_entities(world):
        if name in skip:
            continue
        if inline is not None or name in detailed or name == obj:
            m = inline or sdf.Model(model_dir(uri) / "model.sdf")
            assert not m.meshes, f"{name}: mesh collisions {m.meshes} (box-only scene, unseen)"
            out += [(f"{name}::{c}", T @ m.pose(ln, {}) @ Tc, np.array(s, float))
                    for ln, bs in m.boxes.items() for c, Tc, s in bs]
        else:
            lo, hi = np.array(json.loads((model_dir(uri) / "manifest.json").read_text())["aabb_model_frame"])
            out.append((f"{name}::aabb", T @ sdf.tf((lo + hi) / 2), hi - lo))
    return out


def sd_box(p, T, size):
    """Signed distance of points p (N, 3) to a box (pose T, size)."""
    q = np.abs((p - T[:3, 3]) @ T[:3, :3]) - size / 2
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)


def chassis_size(chassis, offset_m, offset_deg):
    """The chassis box (sx, sy, sz) grown to cover itself moved up to offset_m and turned up to offset_deg."""
    sx, sy, sz = chassis
    t = math.radians(offset_deg)
    hx, hy = sx / 2, sy / 2
    return np.array([2 * (hx * math.cos(t) + hy * math.sin(t) + offset_m),
                     2 * (hy * math.cos(t) + hx * math.sin(t) + offset_m), sz])


def surface(size, spacing=SPACING_M):
    """(points (N, 3) on a box's surface, centred at 0, grid step <= spacing; the largest step)."""
    n = [max(int(math.ceil(s / spacing)), 1) for s in size]
    axes = [np.linspace(-s / 2, s / 2, k + 1) for s, k in zip(size, n)]
    pts = []
    for i in range(3):                  # the two faces normal to axis i, a grid over the other two
        a, b = [axes[j] for j in range(3) if j != i]
        g = np.stack(np.meshgrid(a, b, indexing="ij"), -1).reshape(-1, 2)
        for side in (-size[i] / 2, size[i] / 2):
            pts.append(np.insert(g, i, side, axis=1))
    return np.unique(np.vstack(pts), axis=0), max(s / k for s, k in zip(size, n))


class Clearance:
    """reject(layout) -> None or the reason: the grown chassis at the layout's spot closer than min_m
    to a scene box. chassis: (size x, size y, mount z) [m], the box from 5 mm above the floor up to mount z."""

    def __init__(self, boxes, obj_pose, box_centre, min_m, offset_m, offset_deg, chassis,
                 facing=start_design.FACING_YAW_OBJ):
        self.boxes, self.obj_pose, self.box_centre, self.min_m, self.facing = boxes, obj_pose, box_centre, min_m, facing
        sx, sy, top = chassis
        self.size = chassis_size((sx, sy, top - 0.005), offset_m, offset_deg)
        self.z = (top + 0.005) / 2                          # box centre
        self.pts, step = surface(self.size)
        self.err = step * math.sqrt(2) / 2                 # a face point to its nearest sample

    def pose(self, layout):
        """4x4 world pose of the chassis box at the layout's spot."""
        x, y, yaw = frames.base_to_world(start_design.layout_spot(layout, self.box_centre, self.facing), self.obj_pose)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, self.z], [0, 0, 0, 1.0]])

    def distances(self, layout):
        """{box name: lower bound of the chassis distance [m]} of the boxes that could be within min_m."""
        T = self.pose(layout)
        p = self.pts @ T[:3, :3].T + T[:3, 3]
        r = np.linalg.norm(self.size) / 2
        out = {}
        for name, Tb, size in self.boxes:
            if np.linalg.norm(Tb[:3, 3] - T[:3, 3]) - r - np.linalg.norm(size) / 2 > self.min_m + self.err:
                continue                                    # bounding spheres apart
            inside = sd_box(Tb[None, :3, 3], T, self.size)[0]     # a box wholly inside the chassis
            out[name] = float(min(sd_box(p, Tb, size).min() - self.err, inside))
        return out

    def __call__(self, layout):
        d = self.distances(layout)
        name = min(d, key=d.get, default=None)
        if name is None or d[name] >= self.min_m:
            return None
        return f"collision: chassis {d[name]:.3f} m from {name} (min {self.min_m} m)"


def clearance_reject(task, robot, starts=None):
    """The Clearance of a task (specs/tasks/<task>) for a robot: its scene, object pose, the robot's start-box
    centre (robots/<implementation>/planner.py), chassis and mount, replay.spawn_clearance_min_m and its
    starts block (generator.yaml unless given)."""
    from datagen.plugins.randomisers.robot_base import box_centre_obj
    r = specs.load(specs_bridge.SPECS / "tasks" / task / "task.yaml", robot)
    rb, sc, st = r.robot, r.scene, starts or specs_bridge.round_blocks(task, robot)["starts"]
    boxes = scene_boxes(specs_bridge.path(sc.world), lambda uri: specs_bridge.model_dir(sc.assets, uri),
                        sc.skip, sc.detailed, r.task.object.name)
    return Clearance(boxes, list(r.task.object.world_pose), box_centre_obj(rb.implementation)[task],
                     r.task.replay.spawn_clearance_min_m, st["offset_m"], st["offset_deg"],
                     (*rb.chassis.size_xy, rb.mount.xyz[2]))
