"""gz-sim Python system: Panda pick-and-place controller (the Panda-only can-on-plate demo).

Loaded by gz-sim-python-system-loader-system (see can_demo/can_on_plate.sdf),
so it runs inside the simulation loop, every 1 ms physics step. Control is
in lockstep with MuJoCo: no transport latency, same result on every run.

- Arm: joint torques (the mobile Panda's law, robots/mobile_panda/robot.py): P on the planned trajectory,
  gravity compensation (link masses from franka_description) and feed-forward of the joint damping. The
  D term lives in the SDF joint damping, which MuJoCo integrates implicitly;
  an explicit D gain went unstable near q2 = 0, where joints 1 and 3 align. Velocity commands are avoided on
  purpose: the plugin implements them as a huge implicit damping term that
  MuJoCo's contact solver doesn't see, which pumps momentum into a grasped
  object and throws it out of the gripper.
- Fingers: force-controlled. Each squeezes with a PD capped at 12 N, and a
  stiff common-mode PD keeps the pair centred on the hand, like the Franka
  hand's mechanically coupled fingers.

Progress and the final verdict are published on /pick_place/status.
"""
import math
import os

import numpy as np
from gz.msgs.stringmsg_pb2 import StringMsg
from gz.sim import Joint, Link, Model, World
from gz.transport import Node

import panda_plan as pp
from robots.mobile_panda import robot as panda

START_DELAY = 1.0          # [s] let everything settle before moving
FINGER_FMAX = float(os.environ.get("FINGER_FMAX", 12.0))


def tilt_deg(rot):
    """Angle between a body's z axis and the world z axis."""
    z = rot.rotate_vector(_UNIT_Z)
    return math.degrees(math.acos(max(-1.0, min(1.0, z.z()))))


def xyz(pose):
    p = pose.pos()
    return np.array([p.x(), p.y(), p.z()])


_UNIT_Z = None


def arm_torques(q, q_des, qd_des):
    """Arm joint torques: P + damping feed-forward + gravity compensation (1 ms law)."""
    return pp.PANDA.arm_torques(q, q_des, qd_des)


def finger_forces(fq, fv, grip, fmax=None):
    """Coupled-finger forces (left, right) for per-finger opening target grip."""
    return panda.finger_forces(fq, fv, grip, FINGER_FMAX if fmax is None else fmax)


class PickPlace:
    def configure(self, entity, sdf, ecm, event_mgr):
        global _UNIT_Z
        from gz.math import Vector3d
        _UNIT_Z = Vector3d(0, 0, 1)

        self.ts, self.qs, self.grips, self.events, worst = pp.plan()
        self.qd = np.gradient(self.qs, pp.DT, axis=0)

        world = World(entity)
        arm = Model(world.model_by_name(ecm, "panda"))
        self.arm = [Joint(arm.joint_by_name(ecm, n)) for n in pp.PANDA.arm_joints]
        self.fingers = [Joint(arm.joint_by_name(ecm, n)) for n in pp.PANDA.gripper.joints]
        for j, q0 in zip(self.arm + self.fingers, list(pp.READY) + [0.04, 0.04]):
            j.enable_position_check(ecm)
            j.enable_velocity_check(ecm)
            j.reset_position(ecm, [float(q0)])
        self.hand = Link(arm.link_by_name(ecm, "panda_hand"))
        self.rf = Link(arm.link_by_name(ecm, "panda_rightfinger"))
        self.can = Link(Model(world.model_by_name(ecm, "can")).canonical_link(ecm))
        self.plate = Link(Model(world.model_by_name(ecm, "plate")).canonical_link(ecm))

        self.node = Node()
        self.pub = self.node.advertise("/pick_place/status", StringMsg)
        self.next_event = 0
        self.finished = False
        self.say(f"planned {len(self.ts)} setpoints ({self.ts[-1]:.1f} s), worst IK "
                 f"error {worst[0]*1000:.2f} mm / {math.degrees(worst[1]):.3f} deg")

    def say(self, text):
        print(f"[pick_place] {text}", flush=True)
        msg = StringMsg()
        msg.data = text
        self.pub.publish(msg)

    def setpoint(self, t):
        """Linearly interpolated joint setpoint, velocity and gripper target."""
        x = min(max(t / pp.DT, 0.0), len(self.ts) - 1.0)
        i = min(int(x), len(self.ts) - 2)
        a = x - i
        q = (1 - a) * self.qs[i] + a * self.qs[i + 1]
        qd = (1 - a) * self.qd[i] + a * self.qd[i + 1] if 0 < t < self.ts[-1] else 0 * q
        return q, qd, (1 - a) * self.grips[i] + a * self.grips[i + 1]

    def pre_update(self, info, ecm):
        if info.paused:
            return
        t = info.sim_time.total_seconds() - START_DELAY
        q_des, qd_des, grip = self.setpoint(t)

        q = np.array([(j.position(ecm) or [0.0])[0] for j in self.arm])
        for j, u in zip(self.arm, arm_torques(q, q_des, qd_des)):
            j.set_force(ecm, [float(u)])
        fq = [(j.position(ecm) or [0.0])[0] for j in self.fingers]
        fv = [(j.velocity(ecm) or [0.0])[0] for j in self.fingers]
        for j, u in zip(self.fingers, finger_forces(fq, fv, grip)):
            j.set_force(ecm, [u])

        while self.next_event < len(self.events) and t >= self.events[self.next_event][0]:
            can = xyz(self.can.world_pose(ecm))
            tcp = xyz(self.hand.world_pose(ecm))
            fq = [j.position(ecm)[0] * 1000 for j in self.fingers]
            self.say(f"t={info.sim_time.total_seconds():6.2f}s  "
                     f"{self.events[self.next_event][1]:<18} can {np.round(can, 3)}  "
                     f"hand {np.round(tcp, 3)}  fingers {fq[0]:.1f}/{fq[1]:.1f} mm")
            self.next_event += 1

        if not self.finished and t > self.ts[-1]:
            self.finished = True
            self.report(ecm)

    def report(self, ecm):
        can_pose, plate_pose = self.can.world_pose(ecm), self.plate.world_pose(ecm)
        can, plate = xyz(can_pose), xyz(plate_pose)
        d = np.linalg.norm(can[:2] - plate[:2])
        gap = can[2] - pp.CAN_R - (plate[2] + 0.0075)
        tilt = tilt_deg(can_pose.rot())      # lying on its side = 90 deg
        ok = d < pp.PLATE_R - 0.03 and abs(gap) < 0.01 and abs(tilt - 90) < 10
        self.say(f"DONE {'SUCCESS' if ok else 'FAILED'}: can {np.round(can, 3)}, "
                 f"{d*100:.1f} cm from plate centre, {gap*1000:.1f} mm above plate, "
                 f"tilt {tilt:.1f} deg")


def get_system():
    return PickPlace()
