"""The two experts, ported from main's T4 reference clients (components/simulator/ros/revolve_sim/test/check_open_door.py,
check_open_drawer.py): scripted ramped base segments to a stand-off pose, MoveIt OMPL to a pre-grasp, Pilz LIN onto the
handle bar, small LINs until the TCP is within a mm of its target, a squeeze past closed, then the task's open motion.

The grasp frames come from the scene's model SDFs (the instance's poses), the base pose from the gz ground truth, so the
only thing that varies between scene instances, the robot start, enters through the drive: it starts from wherever the
robot is. What the references assumed about the start (the arm folded) is made so first. Constants are the references'
(measured there; DATAGEN-v2.md 12)."""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from . import motion
from .geometry import back_off, grasp_pose, rotation_about_z, stand_off, tcp_frame
from . import resume as R
from .robot_io import EpisodeOver, Stopped
from .setup import Setup

PRE = 0.10                          # pre-grasp: back along the approach [m]
CENTRE_MM, CENTRE_TRIES = 1.0, 4    # the TCP this close to its target (handle frame) before the squeeze, tries
SETTLE_S = 0.2                      # a resume's first command: hold the measured pose this long


class Expert:
    """One episode: `run()` drives the sim's robot through the task from the scene's start and returns the notes for the datagen
    log; `resume()` takes the task over from wherever the robot is. A task is `drive`, `reach`, `squeeze`, `open`."""
    folded_tol = 0.05               # [rad]: the arm counts as folded within this

    def __init__(self, s: Setup):
        self.s, self.io, self.planner, self.log = s, s.io, s.planner, []
        self.T_h = s.T_handle       # the handle at the closed door / drawer
        self.T_cur = self.T_h       # the handle the reach plans against (the moved one after a resume with the door ajar)
        self.T_b = None             # the base pose after the drive
        self.skip_link = None       # a link left out of the first planning scene (a start at the bar would collide)
        self.t0 = 0.0

    def run(self) -> str:
        self.io.begin_episode()
        self.t0 = self.s.io.client.now()
        try:
            self.fold_arm()
            self.steps()
        except EpisodeOver:
            self.mark("episode_over")
        return json.dumps(self.log)

    def steps(self) -> None:
        self.drive()
        self.reach()
        self.squeeze()
        self.open()

    def drive(self) -> None:
        raise NotImplementedError

    def reach(self) -> None:
        raise NotImplementedError

    def open(self) -> None:
        raise NotImplementedError

    # --- resuming
    def resume(self, state: R.Situation | None = None, stop=None) -> str:
        """Take the task over from the robot's current state (`state` for tests; live otherwise): hold the measured pose, then the
        entry the state calls for and what follows it. Logs `max_delta_action` of the first second. `stop()` (polled with the
        episode's result, so between and during the motion segments) ends it cleanly: base stopped, measured pose held (a closed
        hand keeps squeezing), mark `stopped`. -> the notes, as `run()`."""
        self.io.begin_episode()
        self.io.stop = stop or (lambda: False)
        self.log, self.t0 = [], self.io.client.now()
        self.T_cur, self.T_b, self.skip_link = self.T_h, None, None
        self.io.record_arm(1.0)
        try:
            s = state or self.observe()
            entry = R.entry_for(s, self.s.case)
            self.mark("resume", entry=entry, ajar=R.is_ajar(s) and entry != R.OPEN, **vars(s))
            self.settle()
            if entry != R.OPEN and R.is_ajar(s):
                self.T_cur = self.handle_now()
            if entry != R.APPROACH:
                self.T_b = self.io.base()[0]
            if entry == R.GRASP:
                self.skip_link = self.s.handle.name
            if entry != R.OPEN and R.gripper_closed(s.gripper, self.s.case):
                self.mark("open_hand", code=self.io.move_hand(self.s.case["hand_open"], 0.5))
            if entry == R.APPROACH:
                self.fold_arm()
                self.drive()
            if entry != R.OPEN:
                self.reach()
                self.squeeze()
            self.open()
        except EpisodeOver:
            self.mark("episode_over")
        except Stopped:
            self.mark("stopped")
            self.settle()
        finally:
            self.io.stop = lambda: False
            self.mark("first_second", max_delta_action=R.max_delta_action(self.io.arm_trace, self.t0))
        return json.dumps(self.log)

    def observe(self) -> R.Situation:
        io, case = self.io, self.s.case
        io.wait_for(lambda: case["hand_joint"] in io.joints(), 10)
        outside = np.maximum(np.abs(self.tcp_in_handle()) / 1000 - np.asarray(self.s.handle_box.size) / 2, 0.0)[:3]
        return R.Situation(round(float(np.linalg.norm(outside)), 4), round(io.joints()[case["hand_joint"]], 4),
                           round(io.articulation or 0.0, 4), io.phase())

    def settle(self) -> None:
        """The first commands are the measured arm joints and hand (a closed hand keeps squeezing): no jump from what a policy left."""
        now, case = self.io.joints(), self.s.case
        self.mark("hold", arm=self.io.move_arm([now[j] for j in case["arm_joints"]], SETTLE_S))
        hand = now[case["hand_joint"]]
        self.io.move_hand(case["hand_squeeze"] if R.gripper_closed(hand, case) else hand, SETTLE_S)

    def mark(self, what: str, **kw) -> None:
        self.log.append({"t": round(self.io.client.now() - self.t0, 2), "what": what, "phase": self.io.phase(), **kw})
        print(json.dumps(self.log[-1]), file=sys.stderr, flush=True)

    # --- shared steps
    def fold_arm(self) -> None:
        """The references drive with the arm folded (it stays clear of the door / counter): fold it if the instance did not."""
        q = self.s.robot.poses["folded"]
        now = self.io.joints()
        if max(abs(now[j] - t) for j, t in zip(self.s.case["arm_joints"], q)) > self.folded_tol:
            self.mark("fold", code=self.io.move_arm(q, 3.0))

    def tcp_world(self, T_b, frame: str | None = None) -> np.ndarray:
        """A frame's origin (default the TCP) in the world: tf in the base frame x the ground-truth base pose."""
        return T_b @ self.io.point(frame or self.s.robot.frames["tcp"], self.s.robot.frames["base"])

    def tcp_in_handle(self) -> np.ndarray:
        """The TCP in the handle's frame at its current articulation [mm]."""
        p = np.linalg.inv(self.handle_now()) @ self.tcp_world(self.io.base()[0])
        return np.round(1000 * p[:3], 1)

    def handle_now(self) -> np.ndarray:
        raise NotImplementedError

    def approach(self, T_b, T_grasp, seed, axes, goal_mm, lin_scale=None) -> bool:
        """OMPL to the pre-grasp (IK seeded with `seed`), LIN onto the bar, then LINs until the TCP's `axes` (handle frame)
        are at `goal_mm` within CENTRE_MM (the effort JTC's steady-state error is a few mm; the stiff Panda hand does not
        self-centre, the SO-101's fixed jaw must press the bar). -> the last LIN succeeded."""
        T_inv, T_pre, axes = np.linalg.inv(T_b), back_off(T_grasp, PRE), list(axes)
        self.planner.scene_boxes(self.s.boxes, T_b, skip_link=self.skip_link)
        found, g = self.planner.ik(seed, T_inv @ T_pre)
        self.mark("ik_pre", ok=found)
        pre_ok = found and self.planner.run("ompl_rrtc", goal=g)
        self.io.wait(0.3)
        e = self.tcp_in_handle() - np.round(1000 * (np.linalg.inv(self.T_cur) @ T_pre)[:3, 3], 1)
        self.mark("pre_grasp", ok=pre_ok, err_mm=e.tolist())
        T_cmd, step = T_grasp.copy(), self.T_cur[:3, axes]
        T_cmd[:3, 3] -= step @ e[axes] / 1000.0
        self.planner.scene_boxes(self.s.boxes, T_b, skip_link=self.s.handle.name)      # the bar goes between the fingers
        ok = pre_ok and self.planner.run("pilz_lin", pose=T_inv @ T_cmd, scale=lin_scale)
        for _ in range(CENTRE_TRIES):
            self.io.wait(0.3)
            e = self.tcp_in_handle()
            err = e[axes] - goal_mm
            self.mark("at_bar", ok=ok, tcp_mm=e.tolist())
            if not ok or max(abs(err)) <= CENTRE_MM:
                break
            T_cmd[:3, 3] -= step @ err / 1000.0
            ok = self.planner.run("pilz_lin", pose=T_inv @ T_cmd, scale=lin_scale)
        return ok

    def squeeze(self) -> None:
        """Close the hand past the bar (the effort JTC squeezes at its clamp) until the evaluator leaves `grasp`."""
        self.io.move_hand(self.s.case["hand_squeeze"], 0.5)
        self.io.wait_for(lambda: self.io.phase() not in ("approach", "grasp"), 8)
        self.io.check()
        self.mark("grasped", tcp_mm=self.tcp_in_handle().tolist())

    def hold(self) -> None:
        """Hold what the open motion reached until the evaluator's verdict (the controllers keep their references)."""
        try:
            self.io.wait_for(lambda: self.io.check() or False, 15)
        except EpisodeOver:
            self.mark("result")


class OpenFridgeDoor(Expert):
    """mobile_panda + fridge: Pilz CIRC about the hinge for the first 20 deg, then the base orbits the hinge (one constant
    body twist, the robot turns rigidly with the door) up to 70 deg."""
    DIST, SIDE, YAW_OFF = 0.50, 0.40, math.radians(60)    # stand-off from the handle (found by an IK search over base poses:
    #   >= 0.3 rad to every joint limit, also along the arm's arc); the approach along +y needs the wrist turned
    SEED = [0.187, -1.429, 1.452, -2.148, -0.017, 3.017, 2.188]     # the IK branch
    ARC_ARM, ARC_TOTAL, ARC_MIN = math.radians(20.0), math.radians(70.0), math.radians(1.0)
    V_DRIVE, W_DRIVE, W_ORBIT = 0.3, 0.5, 0.12          # 0.2 rad/s orbit caused finger contact gaps
    CIRC_SCALE = 0.1

    def __init__(self, s):
        super().__init__(s)
        self.hinge, axis = s.articulation[1].T[:3, 3], s.articulation[1].axis
        self.sign = 1.0 if axis[2] > 0 else -1.0

    def door(self, theta: float) -> np.ndarray:
        return rotation_about_z(self.hinge, self.sign * theta)

    def handle_now(self):
        return self.door(self.io.articulation or 0.0) @ self.T_h

    def grasp_at(self, T_h) -> np.ndarray:
        return grasp_pose(T_h, tcp_frame(T_h, 2))           # TCP at the bar centre, +z into the door, x up

    def drive(self) -> None:
        x, y, yaw = stand_off(self.T_cur, self.DIST, self.SIDE, self.YAW_OFF)
        self.io.drive_to((x, y), yaw, self.V_DRIVE, self.W_DRIVE)
        self.io.wait(0.3)
        self.T_b, pose = self.io.base()
        self.mark("drive_done", base=[round(v, 4) for v in pose])

    def reach(self) -> None:
        self.approach(self.T_b, self.grasp_at(self.T_cur), self.SEED, (0, 1), (0.0, 0.0))

    def open(self) -> None:
        """The arm's arc about the hinge up to ARC_ARM, then the base orbits the hinge up to ARC_TOTAL, from the door's angle."""
        io, T_b, T_grasp = self.io, self.T_b, self.grasp_at(self.T_h)
        angle = io.articulation or 0.0
        center = np.append(np.append(self.hinge[:2], T_grasp[2, 3]), 1.0)
        T_inv = np.linalg.inv(T_b)
        if (arc := self.ARC_ARM - angle) > self.ARC_MIN:
            T_arc = self.door(arc) @ self.door(angle) @ T_grasp
            ok = self.planner.run("pilz_circ", pose=T_inv @ T_arc,
                                  constraints=self.planner.circ_constraint((T_inv @ center)[:3]), scale=self.CIRC_SCALE)
            self.mark("arm_arc", ok=ok)
        T_b, _ = io.base()
        hb = (np.linalg.inv(T_b) @ center)[:2]
        io.segment(motion.orbit(hb, self.sign > 0), self.ARC_TOTAL - max(angle, self.ARC_ARM), self.W_ORBIT)
        self.mark("orbit_done", base=[round(v, 4) for v in io.base()[1]])
        self.hold()


class OpenDrawer(Expert):
    """mobile_so101_xl + drawer: the 5-DoF arm reaches the bar in the vertical plane of its shoulder pan axis, so the grasp
    point is the bar point in front of the shoulder (measured after the drive); the base backs off along the slide."""
    REACH = 0.80                    # base_footprint in front of the bar [m]: the TCP 0.80 m ahead
    SEED = [0.0, -0.2718, 1.2384, -0.9666, 0.0]         # the IK branch: TCP 0.75 m ahead at the bar height, horizontal
    BAR_MARGIN = 0.05               # the grasp point stays this far inside the bar's ends [m]
    V_DRIVE, W_DRIVE = 0.3, 0.5
    PULL, V_PULL = 0.20, 0.05       # the base's back-off along the slide [m, m/s] (the task: >= 0.15 m for 1 s)
    PULL_MIN = 0.005                # a shorter rest of the pull is not driven [m]
    LIN_SCALE = 0.1                 # Pilz LIN scaling (joint acceleration limits near a stretched elbow)
    LIFT = 0.003                    # above the bar centre [m]: the fixed jaw's pad (8 mm under the TCP) 2 mm into the bar
    DEPTH = 0.003                   # the bar centre this far beyond the TCP: a bar centred on the TCP overlaps the fixed
    #                                 jaw's tip pad by ~2.6 mm only and sometimes slips off it in the squeeze

    def __init__(self, s):
        super().__init__(s)
        self.axis = s.articulation[1].axis

    def handle_now(self):
        T = self.T_h.copy()
        T[:3, 3] += (self.io.articulation or 0.0) * self.axis
        return T

    def drive(self) -> None:
        x, y, yaw = stand_off(self.T_cur, self.REACH, 0.0, 0.0)
        self.io.drive_to((x, y), yaw, self.V_DRIVE, self.W_DRIVE)
        self.io.wait(0.3)
        self.T_b, pose = self.io.base()
        self.mark("drive_done", base=[round(v, 4) for v in pose])

    def reach(self) -> None:
        s, T_h = self.s, self.T_cur
        shoulder = self.tcp_world(self.T_b, "shoulder_link")
        half = 0.5 * s.handle_box.size[1] - self.BAR_MARGIN               # the bar runs along its collision y
        along = max(-half, min(half, float((shoulder[:3] - T_h[:3, 3]) @ T_h[:3, 1])))
        T_grasp = grasp_pose(T_h, tcp_frame(T_h, 1), (self.DEPTH, along, 0.0))
        self.mark("grasp_point", along=round(along, 4))
        self.approach(self.T_b, T_grasp, self.SEED, (0, 2), np.array([1000 * self.DEPTH, 1000 * self.LIFT]), self.LIN_SCALE)

    def open(self) -> None:
        """Back the base off along the slide until the drawer has been pulled PULL out."""
        io = self.io
        d = (self.T_b[:3, :3].T @ self.axis)[:2]                         # the slide's direction in the base frame
        if (pull := self.PULL - (io.articulation or 0.0)) > self.PULL_MIN:
            io.segment((*(d / np.linalg.norm(d)), 0.0), pull, self.V_PULL)
        self.mark("pull_done", base=[round(v, 4) for v in io.base()[1]], slide=round(io.articulation or 0.0, 4))
        self.hold()


EXPERTS = {"open_fridge_door": OpenFridgeDoor, "open_drawer": OpenDrawer}
