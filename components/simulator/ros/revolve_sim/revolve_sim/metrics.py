"""The task metrics (revolve_common.specs.METRICS) on a snapshot of the world: plain numpy, no ROS.

A Snapshot holds what the evaluator node gathered from the upstream topics at one instant:
  poses     {(model, link): 4x4 world pose}: scene model links (gz PosePublisher) and robot links (ground-truth base
            pose x robot_state_publisher tf), keyed (robot name, URDF link) for the robot
  stamps    {(model, link): the time its pose is from} where known (the robot links: their tf stamp, which can lag
            the snapshot time by tens of ms under load); peak_body_speed differences poses over these
  joints    {(model, joint): position}: scene joints (gz JointStatePublisher)
  contacts  [(collision a, collision b, depth)]: scoped gz collision names "model::link::collision" (gz Contact sensors)
  commands  count of non-finite command messages so far; gz_lines: the gz error lines so far (text)
Metrics(resolved, robot_links, collisions).value(condition, snap, key) -> float (nan: not computable now).
"""
import math
import re

import numpy as np
from revolve_common import specs

GROUND = 'ground_plane'                  # world.py's ground model: never a contact partner
LUMP = '_fixed_joint_lump__'             # sdformat's name for a collision merged into a parent link
AXIS = {'x': (0, 1), '-x': (0, -1), 'y': (1, 1), '-y': (1, -1), 'z': (2, 1), '-z': (2, -1)}
NAN = float('nan')


# ------------------------------------------------------------------ geometry
def pose_matrix(xyz, quat_xyzw):
    x, y, z, w = quat_xyzw
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = xyz
    return T


def rpy_matrix(x, y, z, roll, pitch, yaw):
    """SDF <pose> x y z roll pitch yaw -> 4x4 (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                              math.sin(yaw))
    T = np.eye(4)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = (x, y, z)
    return T


def inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def point_distance(p_local, geom):
    """Distance from a point (in the geometry's frame) to the geometry (0 inside). geom: (type, params)."""
    kind, g = geom
    if kind == 'box':
        q = np.abs(p_local) - np.asarray(g['size']) / 2.0
        return float(np.linalg.norm(np.maximum(q, 0.0)))
    if kind == 'sphere':
        return max(0.0, float(np.linalg.norm(p_local)) - g['radius'])
    if kind in ('cylinder', 'capsule'):
        r, h = g['radius'], g['length'] / 2.0
        radial, axial = math.hypot(p_local[0], p_local[1]), abs(p_local[2])
        if kind == 'capsule':
            return max(0.0, math.hypot(radial, max(0.0, axial - h)) - r)
        return math.hypot(max(0.0, radial - r), max(0.0, axial - h))
    return float(np.linalg.norm(p_local))   # mesh / other: to the collision's origin (see collision_geometry)


def collision_geometry(sdf_path):
    """{(link, collision): (4x4 pose in the link, (type, params))} of a model SDF's top-level links. Meshes and other
    shapes get type 'origin' (distance to them = to the collision origin)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(sdf_path).getroot()
    model = root.find('model') if root.tag == 'sdf' else root
    out = {}
    for link in model.findall('link'):
        for c in link.findall('collision'):
            pose = c.find('pose')
            if pose is not None and pose.get('relative_to') not in (None, link.get('name')):
                raise specs.SpecError(f'{sdf_path}: collision {c.get("name")}: pose relative_to is not supported')
            vals = [float(v) for v in (pose.text.split() if pose is not None and pose.text else [])] or [0.0] * 6
            g = c.find('geometry')
            shape = g[0] if g is not None and len(g) else None
            kind, params = 'origin', {}
            if shape is not None and shape.tag == 'box':
                kind, params = 'box', {'size': [float(v) for v in shape.findtext('size').split()]}
            elif shape is not None and shape.tag == 'sphere':
                kind, params = 'sphere', {'radius': float(shape.findtext('radius'))}
            elif shape is not None and shape.tag in ('cylinder', 'capsule'):
                kind, params = shape.tag, {'radius': float(shape.findtext('radius')),
                                           'length': float(shape.findtext('length'))}
            out[(link.get('name'), c.get('name'))] = (rpy_matrix(*vals), (kind, params))
    return out


def compare(op, value, threshold):
    if value is None or not math.isfinite(value):
        return False
    return {'lt': value < threshold, 'le': value <= threshold, 'eq': value == threshold, 'ge': value >= threshold,
            'gt': value > threshold}[op]


# ------------------------------------------------------------------ entities
class Ref:
    """A resolved entity: kind robot_links / robot_link / model / link / collision / joint / any / any_except_targets."""

    def __init__(self, kind, model=None, name=None, collision=None, links=()):
        self.kind, self.model, self.name, self.collision, self.links = kind, model, name, collision, tuple(links)

    def matches(self, scoped, robot):
        """Does the scoped gz collision name 'model::link::collision' belong to this entity?"""
        m, link, col = split_collision(scoped)
        if self.kind in ('any', 'any_except_targets'):
            return m not in (robot, GROUND)
        if self.kind in ('robot_link', 'robot_links'):
            return m == robot and robot_link_of(link, col) in self.links
        if self.kind == 'model':
            return m == self.model
        if self.kind == 'link':
            return m == self.model and link == self.name
        if self.kind == 'collision':
            return m == self.model and link == self.name and col == self.collision
        return False


def split_collision(scoped):
    parts = scoped.split('::')
    return (parts + ['', '', ''])[:3] if len(parts) < 3 else (parts[0], parts[-2], parts[-1])


def robot_link_of(gz_link, collision):
    """The URDF link of a robot collision: a collision merged by fixed-joint lumping is named
    '<parent>_fixed_joint_lump__<child>_collision[_N]' (sdformat), otherwise it is the gz link's own."""
    if LUMP in collision:
        return re.sub(r'_collision(_\d+)?$', '', collision.split(LUMP, 1)[1])
    return gz_link


# ------------------------------------------------------------------ snapshot + metrics
class Snapshot:
    def __init__(self, t, poses=None, joints=None, contacts=None, commands=0, gz_lines=(), stamps=None):
        self.t, self.poses, self.joints, self.stamps = t, poses or {}, joints or {}, stamps or {}
        self.contacts, self.commands, self.gz_lines = contacts or [], commands, gz_lines


class Metrics:
    """Computes condition values. resolved: specs.Resolved (with the robot); robot_links: the robot's URDF links;
    collisions: {model entity: collision_geometry(...)}."""

    SPEED_WINDOW_S = 0.02            # peak_body_speed: finite difference over at least this much sim time

    def __init__(self, resolved, robot_links, collisions):
        self.r, self.robot = resolved, resolved.robot
        self.robot_name = resolved.robot.name
        self.robot_links = list(robot_links)
        self.collisions = collisions
        self.targets = [self.ref(t) for t in resolved.task.targets.values()]
        self.slip_ref = {}
        self.history = []            # [(t, {(model, link): (stamp, xyz)})] for peak_body_speed

    def reset_phase(self):
        self.slip_ref.clear()

    def ref(self, text):
        e = specs.parse_entity(text)
        if not e.robot and not e.any and e.name is None and e.model in self.r.task.targets:
            e = specs.parse_entity(self.r.task.targets[e.model])
        if e.any:
            return Ref(e.text)
        if e.robot:
            if e.name is None:
                return Ref('robot_links', links=self.robot_links)
            if e.name in self.robot.frames:
                return Ref('robot_link', links=[self.robot.frames[e.name]])
            return Ref('robot_links', links=self.robot.groups[e.name])
        if e.collision:
            return Ref('collision', e.model, e.name, e.collision)
        if e.name is None:
            return Ref('model', e.model)
        if e.name in self.r.models[e.model].joints:
            return Ref('joint', e.model, e.name)
        return Ref('link', e.model, e.name)

    # poses
    def frame(self, ref, snap):
        """4x4 world pose of a point entity (a collision: its frame), or None."""
        if ref.kind == 'robot_link':
            return snap.poses.get((self.robot_name, ref.links[0]))
        T = snap.poses.get((ref.model, ref.name))
        if T is None or ref.kind == 'link':
            return T
        return T @ self.collisions[ref.model][(ref.name, ref.collision)][0]

    def bodies(self, ref):
        """[(model, link)] of a body entity (for speeds)."""
        if ref.kind in ('robot_link', 'robot_links'):
            return [(self.robot_name, lk) for lk in ref.links]
        if ref.kind == 'model':
            return [(ref.model, lk) for lk in self.r.models[ref.model].links]
        return [(ref.model, ref.name)]

    def value(self, cond, snap, key=None):
        p = cond.params
        f = getattr(self, 'm_' + cond.metric)
        return f(p, snap, key)

    def m_joint(self, p, snap, key):
        r = self.ref(p['of'])
        return snap.joints.get((r.model, r.name), NAN)

    def m_distance(self, p, snap, key):
        a, b = self.ref(p['from']), self.ref(p['to'])
        if a.kind != 'collision' and b.kind == 'collision':
            a, b = b, a                       # the geometry side first
        Ta, Tb = self.frame(a, snap), self.frame(b, snap)
        if Ta is None or Tb is None:
            return NAN
        if a.kind == 'collision':             # the other side's origin to a's geometry
            local = (inv(Ta) @ np.append(Tb[:3, 3], 1.0))[:3]
            return point_distance(local, self.collisions[a.model][(a.name, a.collision)][1])
        return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))

    def m_angle(self, p, snap, key):
        Ta, Tb = self.frame(self.ref(p['from']), snap), self.frame(self.ref(p['to']), snap)
        if Ta is None or Tb is None:
            return NAN
        (ia, sa), (ib, sb) = AXIS[p['axes'][0]], AXIS[p['axes'][1]]
        return float(math.acos(np.clip(sa * sb * Ta[:3, ia] @ Tb[:3, ib], -1.0, 1.0)))

    def in_contact(self, a, b, snap):
        """{robot / model link} of entity a touching entity b."""
        out = set()
        for c1, c2, _ in snap.contacts:
            for x, y in ((c1, c2), (c2, c1)):
                if a.matches(x, self.robot_name) and b.matches(y, self.robot_name) and \
                        split_collision(x)[0] != split_collision(y)[0]:
                    if b.kind == 'any_except_targets' and any(t.matches(y, self.robot_name) for t in self.targets):
                        continue
                    m, lk, col = split_collision(x)
                    out.add((m, robot_link_of(lk, col) if m == self.robot_name else lk))
        return out

    def m_contacts(self, p, snap, key):
        return float(len(self.in_contact(self.ref(p['from']), self.ref(p['to']), snap)))

    def m_gripper_opening(self, p, snap, key):
        links = self.ref(p['of']).links
        Ts = [snap.poses.get((self.robot_name, lk)) for lk in links]
        if len(Ts) != 2 or any(T is None for T in Ts):
            return NAN
        return float(np.linalg.norm(Ts[0][:3, 3] - Ts[1][:3, 3]))

    def m_slip(self, p, snap, key):
        Ta, Tb = self.frame(self.ref(p['from']), snap), self.frame(self.ref(p['to']), snap)
        if Ta is None or Tb is None:
            return NAN
        rel = (inv(Tb) @ np.append(Ta[:3, 3], 1.0))[:3]
        ref = self.slip_ref.setdefault(key, rel)
        return float(np.linalg.norm(rel - ref))

    def record(self, snap):
        """Keep the link positions for peak_body_speed (call once per tick, before value())."""
        self.history.append((snap.t, {k: (snap.stamps.get(k, snap.t), T[:3, 3].copy())
                                      for k, T in snap.poses.items()}))
        while len(self.history) > 2 and self.history[1][0] <= snap.t - self.SPEED_WINDOW_S:
            self.history.pop(0)

    def m_peak_body_speed(self, p, snap, key):
        if len(self.history) < 2:
            return NAN
        old, new = self.history[0][1], self.history[-1][1]
        speeds = [np.linalg.norm(new[b][1] - old[b][1]) / (new[b][0] - old[b][0])
                  for b in self.bodies(self.ref(p['of']))
                  if b in new and b in old and new[b][0] - old[b][0] >= 0.5 * self.SPEED_WINDOW_S]
        return float(max(speeds)) if speeds else NAN

    def m_penetration(self, p, snap, key):
        r = self.ref(p['of'])
        depths = [d for c1, c2, d in snap.contacts if r.matches(c1, self.robot_name) or r.matches(c2, self.robot_name)]
        return float(max(depths, default=0.0))

    def m_non_finite_commands(self, p, snap, key):
        return float(snap.commands)

    def m_gz_log_problem_lines(self, p, snap, key):
        pats = [re.compile(x) for x in p['ignore']]
        return float(sum(1 for ln in snap.gz_lines if not any(r.search(ln) for r in pats)))
