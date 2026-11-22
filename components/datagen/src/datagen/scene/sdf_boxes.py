"""The collision geometry of a scene as world-frame oriented boxes, read from its model SDFs (numpy + stdlib).

Joints are taken at 0 (the scene's initial joint values must be 0 for now, DATAGEN-v2.md 9). Each collision is a box:
`box` exactly, `cylinder` / `sphere` as their bounding box, `mesh` (OBJ) as the AABB of its vertices (never smaller
than the mesh); `plane` is skipped (the ground). Poses: scene model pose (x, y, z, yaw) @ link pose (model frame) @
collision pose (link frame); SDF's R = Rz Ry Rx."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from revolve_common import specs

from datagen.core.spec import SpecError


@dataclass(frozen=True)
class Box:
    name: str                   # <entity>::<link>::<collision>
    pose: np.ndarray            # 4x4 world pose of the box centre
    size: tuple[float, float, float]

    def corners(self) -> np.ndarray:
        """(8, 3) world positions."""
        h = np.array(self.size) / 2
        local = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return local @ self.pose[:3, :3].T + self.pose[:3, 3]


def tf(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    (cr, cp, cy), (sr, sp, sy) = (np.cos(rpy), np.sin(rpy))
    T = np.eye(4)
    T[:3, :3] = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                          [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                          [-sp, cp * sr, cp * cr]])
    T[:3, 3] = xyz
    return T


def _pose(el) -> np.ndarray:
    p = None if el is None else el.find("pose")
    if p is None or not (p.text or "").strip():
        return np.eye(4)
    v = [float(x) for x in p.text.split()]
    if p.get("relative_to"):
        raise SpecError(f"<pose relative_to='{p.get('relative_to')}'>: only link/collision poses in their parent frame")
    return tf(v[:3], v[3:6])


def _obj_aabb(path: Path, scale) -> tuple[np.ndarray, np.ndarray]:
    try:
        lines = path.read_text().splitlines()
        v = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines if ln.startswith("v ")])
    except (OSError, ValueError) as e:
        raise SpecError(f"{path}: cannot read the mesh: {e}") from None
    if not len(v):
        raise SpecError(f"{path}: no vertices")
    v = v * scale
    return v.min(0), v.max(0)


@lru_cache(maxsize=None)
def _mesh_box(path: str, scale: tuple[float, ...]) -> tuple[np.ndarray, tuple]:
    lo, hi = _obj_aabb(Path(path), np.array(scale))
    return tf((lo + hi) / 2), tuple(hi - lo)


def _geometry(col, resolve) -> tuple[np.ndarray, tuple] | None:
    """(pose in the collision frame, size) of a collision's geometry; None for a plane."""
    g = col.find("geometry")
    shape = g[0] if g is not None and len(g) else None
    if shape is None:
        raise SpecError(f"collision {col.get('name')}: no geometry")
    if shape.tag == "box":
        return np.eye(4), tuple(float(x) for x in shape.findtext("size").split())
    if shape.tag == "cylinder":
        r, ln = float(shape.findtext("radius")), float(shape.findtext("length"))
        return np.eye(4), (2 * r, 2 * r, ln)
    if shape.tag == "sphere":
        r = float(shape.findtext("radius"))
        return np.eye(4), (2 * r,) * 3
    if shape.tag == "mesh":
        scale = tuple(float(x) for x in (shape.findtext("scale") or "1 1 1").split())
        return _mesh_box(str(resolve(shape.findtext("uri").strip())), scale)
    if shape.tag == "plane":
        return None
    raise SpecError(f"collision {col.get('name')}: unsupported geometry <{shape.tag}>")


def model_boxes(entity: str, sdf: Path, T: np.ndarray, resolve) -> list[Box]:
    """The boxes of one model SDF placed at world pose T. resolve(uri) -> the file of a model:// uri."""
    try:
        root = ET.parse(sdf).getroot()
    except (OSError, ET.ParseError) as e:
        raise SpecError(f"{sdf}: {e}") from None
    model = root.find("model") if root.tag == "sdf" else root
    out = []
    for link in model.findall("link"):
        TL = T @ _pose(link)
        for col in link.findall("collision"):
            geo = _geometry(col, resolve)
            if geo is not None:
                out.append(Box(f"{entity}::{link.get('name')}::{col.get('name')}", TL @ _pose(col) @ geo[0], geo[1]))
    return out


def scene_boxes(scene: specs.Scene, repo) -> list[Box]:
    """Every collision of every model of the scene (assets under `repo`) as a world-frame box."""
    models, errors = specs.scene_models(scene, repo)
    if errors:
        raise SpecError("\n".join(errors))

    def resolver(uri: str) -> Path:           # model://<name>/<file>
        name, _, rest = uri.removeprefix("model://").partition("/")
        for d in scene.assets:
            if (p := Path(repo) / d / name / rest).is_file():
                return p
        raise SpecError(f"{uri}: in none of {scene.assets} (under {repo})")
    out = []
    for entity, m in scene.models.items():
        x, y, z, yaw = m.pose
        out += model_boxes(entity, models[entity].sdf, tf((x, y, z), (0.0, 0.0, yaw)), resolver)
    return out
