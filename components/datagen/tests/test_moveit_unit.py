"""The moveit generator plugin (plugins/moveit/) without ROS: the pure helpers (ramped base segments, grasp frames, SDF
geometry), the experts' sequences on a fake robot + planner (a bias in every LIN is centred away), the generator's plumbing.
The parts that need rclpy / moveit_py (robot_io, planner) run on the sim: test_moveit_gpu.py."""
import math
from pathlib import Path

import numpy as np
import pytest

from datagen.core import registry
from datagen.core.errors import DatagenError
from datagen.plugins.moveit import experts, geometry as G, motion, resume as R, setup
from datagen.plugins.moveit.generator import EXPERT, MoveItGenerator
from datagen.plugins.moveit.robot_io import EpisodeOver

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "specs" / "scenes" / "kitchen_partnet" / "scene.yaml"
needs_assets = pytest.mark.skipif(not (REPO / "assets/generated_partnet/fridge/model.sdf").is_file(),
                                  reason="needs assets/ (./revolve assets)")


# ------------------------------------------------------------------ motion.py
@pytest.mark.parametrize("amount, rate", [(1.0, 0.3), (0.2, 0.05), (0.05, 0.3), (1.2, 0.5), (0.35, 0.12)])
def test_a_segment_covers_exactly_its_amount_and_never_exceeds_the_rate(amount, rate):
    peak, total = motion.profile(amount, rate)
    ts = np.linspace(0.0, total, 20001)
    v = np.array([motion.ramp(t, total) * peak for t in ts])
    assert abs(np.trapezoid(v, ts) - amount) < 1e-3 * max(1.0, amount) and v.max() <= rate + 1e-9
    assert v[0] == 0.0 and v[-1] == 0.0                  # ramped at both ends (no yaw kick)


def test_a_zero_amount_is_no_segment():
    assert motion.profile(0.0, 0.3) == (0.0, 0.0)


def test_rotation_takes_the_short_way():
    twist, amount, rate = motion.rotation(math.radians(170), math.radians(-170), 0.5)
    assert twist == (0.0, 0.0, 1.0) and amount == pytest.approx(math.radians(20)) and rate == 0.5
    assert motion.rotation(0.0, -0.3, 0.5)[0] == (0.0, 0.0, -1.0)


def test_translation_is_a_strafe_in_the_body_frame():
    twist, dist, _ = motion.translation((1.0, 1.0, math.pi / 2), (1.0, 3.0), 0.3)     # 2 m ahead of a base facing +y
    assert twist == pytest.approx((1.0, 0.0, 0.0)) and dist == pytest.approx(2.0)
    twist, dist, _ = motion.translation((0.0, 0.0, math.pi / 2), (2.0, 0.0), 0.3)     # 2 m to its right
    assert twist == pytest.approx((0.0, -1.0, 0.0)) and dist == pytest.approx(2.0)
    assert motion.translation((0.0, 0.0, 0.0), (0.0, 0.0), 0.3)[1:] == (0.0, 0.3)


def test_the_orbit_twist_turns_the_robot_rigidly_about_the_hinge():
    hinge = np.array([0.6, -0.4])
    vx, vy, w = motion.orbit(hinge, ccw=True)
    # the velocity of the point at the hinge, v + w x r, is zero
    assert (vx - w * hinge[1], vy + w * hinge[0]) == pytest.approx((0.0, 0.0))
    assert motion.orbit(hinge, ccw=False) == tuple(-c for c in (vx, vy, w))


# ------------------------------------------------------------------ geometry.py
def test_quaternion_of_a_rotation_matrix():
    for yaw in (-2.5, 0.0, 0.7, 3.0):
        x, y, z, w = G.quat(G.rpy_matrix(0, 0, 0, 0.3, -0.2, yaw)[:3, :3])
        assert x * x + y * y + z * z + w * w == pytest.approx(1.0)
    assert G.quat(np.eye(3)) == (0.0, 0.0, 0.0, 1.0)
    assert G.quat(G.rpy_matrix(0, 0, 0, 0, 0, math.pi / 2)[:3, :3])[2:] == pytest.approx((math.sin(math.pi / 4),) * 2)


def handle_frame(x_out=(0.0, -1.0, 0.0), centre=(-1.787, 2.352, 1.025)):
    """A handle frame as the fridge's: x out of the front, z up, y = z x x."""
    T = np.eye(4)
    z = np.array([0.0, 0.0, 1.0])
    T[:3, 0], T[:3, 2], T[:3, 1], T[:3, 3] = x_out, z, np.cross(z, x_out), centre
    return T


def test_tcp_frame_points_into_the_handle_with_x_along_the_bar():
    T_h = handle_frame()
    for along in (1, 2):
        R = G.tcp_frame(T_h, along)
        assert R.T @ R == pytest.approx(np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
        assert R[:, 2] == pytest.approx(-T_h[:3, 0]) and R[:, 0] == pytest.approx(T_h[:3, along])


def test_grasp_pose_and_pre_grasp():
    T_h = handle_frame()
    T = G.grasp_pose(T_h, G.tcp_frame(T_h, 2), offset=(0.003, 0.1, 0.0))
    assert T[:3, 3] == pytest.approx(T_h[:3, 3] + 0.003 * T_h[:3, 0] + 0.1 * T_h[:3, 1])
    pre = G.back_off(T, 0.1)
    assert (pre[:3, 3] - T[:3, 3]) == pytest.approx(0.1 * T_h[:3, 0]) and (pre[:3, :3] == T[:3, :3]).all()


def test_stand_off_is_the_references_world_pose_for_the_fridge_handle():
    # check_open_door.py: the base DIST 0.50 in front of the handle, SIDE 0.40 towards the hinge (+x), yaw 30 deg
    x, y, yaw = G.stand_off(handle_frame(), 0.50, 0.40, math.radians(60))
    assert (x, y, yaw) == pytest.approx((-1.787 + 0.40, 2.352 - 0.50, math.radians(30)))
    x, y, yaw = G.stand_off(handle_frame(), 0.80, 0.0, 0.0)           # the drawer: straight in front, facing it
    assert (x, y, yaw) == pytest.approx((-1.787, 2.352 - 0.80, math.pi / 2))
    assert G.wrap(G.stand_off(handle_frame((-1.0, 0.0, 0.0)), 1.0, 0.0, 0.0)[2]) == pytest.approx(0.0)    # facing +x


def test_rotation_about_a_vertical_line_fixes_that_line():
    pivot = np.array([1.0, 2.0, 0.5])
    M = G.rotation_about_z(pivot, math.pi / 2)
    assert (M @ np.append(pivot, 1.0))[:3] == pytest.approx(pivot)
    assert (M @ np.array([2.0, 2.0, 0.5, 1.0]))[:3] == pytest.approx([1.0, 3.0, 0.5])
    assert G.rotation_about_z(pivot, -0.4) @ G.rotation_about_z(pivot, 0.4) == pytest.approx(np.eye(4))


def test_model_geometry_boxes_and_joints_in_the_world(tmp_path):
    sdf = tmp_path / "model.sdf"
    sdf.write_text("""<sdf version="1.9"><model name="m">
      <link name="body"><pose>0 0 1 0 0 0</pose></link>
      <link name="door"><pose>0.5 0 0 0 0 0</pose>
        <collision name="bar"><pose>0 0.1 0 0 0 0</pose><geometry><box><size>0.01 0.02 0.3</size></box></geometry></collision>
        <collision name="ball"><geometry><sphere><radius>0.1</radius></sphere></geometry></collision></link>
      <joint name="hinge" type="revolute"><parent>body</parent><child>door</child><pose>0.2 0 0 0 0 0</pose>
        <axis><xyz>0 0 1</xyz></axis></joint>
      <joint name="fixed" type="fixed"><parent>body</parent><child>door</child></joint></model></sdf>""")
    g = G.model_geometry([2.0, 3.0, 0.0, math.pi / 2], sdf)           # the model turned 90 deg, at (2, 3)
    assert [(b.link, b.name, b.size) for b in g.boxes] == [("door", "bar", [0.01, 0.02, 0.3])]    # boxes only
    # door link: model (2, 3, 0) yaw 90 deg + (0.5, 0, 0) in the model = (2, 3.5, 0); the bar 0.1 along the model's y = -x
    assert g.boxes[0].T[:3, 3] == pytest.approx([1.9, 3.5, 0.0])
    j = g.joints["hinge"]
    assert set(g.joints) == {"hinge"} and j.child == "door" and j.axis == pytest.approx([0, 0, 1])
    assert j.T[:3, 3] == pytest.approx([2.0, 3.7, 0.0])               # +0.2 along the door's x = the world's y


@needs_assets
def test_the_scene_models_geometry_gives_the_references_handle_and_hinge():
    from revolve_common import specs
    scene = specs.load(BASE)
    models, _ = specs.scene_models(scene, REPO)
    kw = setup.load(BASE, "open_fridge_door", "mobile_panda", REPO)
    S = setup.Setup(io=None, planner=None, **kw)
    assert S.T_handle[:3, 3] == pytest.approx([-1.787, 2.352, 1.025], abs=2e-3)
    assert S.T_handle[:3, 0] == pytest.approx([0, -1, 0], abs=1e-4) and S.handle_box.size == [0.012, 0.014, 0.25]
    joint, j = S.articulation
    assert joint == "top_door_hinge" and j.child == "top_door" and j.axis[2] > 0
    assert G.stand_off(S.T_handle, 0.5, 0.4, math.radians(60)) == pytest.approx((-1.387, 1.852, math.radians(30)), abs=2e-3)
    d = setup.load(BASE, "open_drawer", "mobile_so101_xl", REPO)
    assert d["articulation"][0] == "middle_slide_top" and d["handle"].name == "drawer2_top"


def test_setup_refuses_a_robot_without_a_controller_table():
    with pytest.raises(ValueError, match="no controller table for robot 'mobile_so101'"):
        setup.load(BASE, "open_drawer", "mobile_so101", REPO)


# ------------------------------------------------------------------ the experts on a fake robot + planner
class FakeClient:
    results = {}

    def now(self):
        return 1.0


class FakeRobot:
    """RobotIO + ArmPlanner of a robot that lands every LIN `bias` m (world x) off its target: a TCP at `tcp` (world)."""

    def __init__(self, base, bias, arm_error=0.0):
        self.client, self.articulation, self.pose, self.bias, self.calls = FakeClient(), None, base, bias, []
        self.tcp, self.arm_error, self.over_after = np.array([0.0, 0.0, 1.0, 1.0]), arm_error, None
        self.hand, self.arm_trace = 0.04, []

    # RobotIO
    def begin_episode(self): ...
    def check(self):
        if self.over_after is not None and len(self.calls) >= self.over_after:
            raise EpisodeOver()
    def phase(self): return "approach"
    def wait(self, s): self.check()
    def wait_for(self, pred, wall_s): return True
    def joints(self):
        q = {j: v + self.arm_error for j, v in zip(self.case["arm_joints"], self.folded)}
        return {**q, self.case["hand_joint"]: self.hand}
    def record_arm(self, window_s): ...
    def base(self):
        x, y, yaw = self.pose
        return G.rpy_matrix(x, y, 0, 0, 0, yaw), self.pose
    def point(self, frame, in_frame):
        return np.linalg.inv(self.base()[0]) @ (self.tcp if frame != "shoulder_link" else self.shoulder)
    def drive_to(self, xy, yaw, v, w):
        self.calls.append(("drive_to", tuple(round(c, 3) for c in xy), round(yaw, 3)))
        self.pose = (*xy, yaw)
    def segment(self, direction, amount, rate): self.calls.append(("segment", tuple(np.round(direction, 3)), amount, rate))
    def move_arm(self, q, duration): self.calls.append(("arm", duration)); return 0
    def move_hand(self, q, duration): self.calls.append(("hand", q)); return 0

    # ArmPlanner
    def scene_boxes(self, boxes, T_base, skip_link=None): self.calls.append(("scene", skip_link))
    def ik(self, seed, T):
        self.calls.append(("ik",))
        return True, ("goal", self.base()[0] @ T)
    def run(self, planner, goal=None, pose=None, constraints=None, scale=None):
        self.calls.append(("plan", planner, scale))
        T = goal[1] if goal is not None else self.base()[0] @ pose
        if planner != "pilz_circ":
            self.tcp = np.append(T[:3, 3] + (self.bias if planner == "pilz_lin" else 0.0), 1.0)
        return True
    def circ_constraint(self, center): return center


def fake_robot(task, robot, **kw):
    from revolve_common import specs
    S = setup.Setup(io=None, planner=None, **setup.load(BASE, task, robot, REPO))
    f = FakeRobot((-1.75, 1.1, math.pi / 2), np.array(kw.pop("bias", [0.002, 0.0, 0.0])), **kw)
    f.case, f.folded = S.case, S.robot.poses["folded"]
    f.shoulder = np.array([-2.0, 0.29, 0.9, 1.0])
    S.io = S.planner = f
    return S, f


def plans(f):
    return [c[1] for c in f.calls if c[0] == "plan"]


@needs_assets
def test_the_fridge_expert_drives_to_the_stand_off_centres_the_tcp_squeezes_and_orbits_the_hinge():
    S, f = fake_robot("open_fridge_door", "mobile_panda")
    notes = experts.OpenFridgeDoor(S).run()
    assert f.calls[0] == ("drive_to", (-1.387, 1.852), round(math.radians(30), 3))          # the reference's base pose
    assert plans(f) == ["ompl_rrtc", "pilz_lin", "pilz_lin", "pilz_circ"]                    # one centring LIN
    assert ("hand", S.case["hand_squeeze"]) in f.calls and [c for c in f.calls if c[0] == "arm"] == []   # folded: no fold
    hit = np.linalg.inv(S.T_handle) @ f.tcp
    assert hit[:2] == pytest.approx([0.0, 0.0], abs=experts.CENTRE_MM / 1000)               # within a mm of the bar centre
    seg = f.calls[-1]                                                                        # the orbit, one constant twist
    assert seg[0] == "segment" and seg[3] == experts.OpenFridgeDoor.W_ORBIT
    assert seg[2] == pytest.approx(math.radians(50)) and seg[1][2] == 1.0                    # 70 - 20 deg, ccw
    assert '"what": "orbit_done"' in notes


@needs_assets
def test_the_fridge_expert_folds_an_arm_that_the_instance_left_unfolded():
    S, f = fake_robot("open_fridge_door", "mobile_panda", arm_error=0.3)
    experts.OpenFridgeDoor(S).run()
    assert f.calls[0] == ("arm", 3.0) and f.calls[1][0] == "drive_to"


@needs_assets
def test_the_drawer_expert_grasps_in_front_of_the_shoulder_and_backs_off_along_the_slide():
    S, f = fake_robot("open_drawer", "mobile_so101_xl", bias=[0.0, 0.0, 0.005])
    notes = experts.OpenDrawer(S).run()
    h = S.T_handle[:3, 3]
    assert f.calls[0][0] == "drive_to" and f.calls[0][1] == pytest.approx((h[0] + 0.8, h[1]), abs=1e-3)
    assert plans(f)[:2] == ["ompl_rrtc", "pilz_lin"] and set(plans(f)[1:]) == {"pilz_lin"}
    assert ("plan", "pilz_lin", experts.OpenDrawer.LIN_SCALE) in f.calls
    err = np.linalg.inv(S.T_handle) @ f.tcp
    assert err[[0, 2]] * 1000 == pytest.approx([3.0, 3.0], abs=experts.CENTRE_MM)           # DEPTH and LIFT
    kind, direction, amount, rate = f.calls[-1]
    assert (kind, amount, rate) == ("segment", 0.2, 0.05) and direction[0] == pytest.approx(-1.0, abs=1e-3)   # backwards
    assert '"what": "grasp_point"' in notes and ("hand", S.case["hand_squeeze"]) in f.calls


@needs_assets
def test_an_episode_the_evaluator_ended_stops_the_expert_where_it_is():
    S, f = fake_robot("open_fridge_door", "mobile_panda")
    f.over_after = 3
    notes = experts.OpenFridgeDoor(S).run()
    assert notes.count("episode_over") == 1 and '"orbit_done"' not in notes and ("hand", -0.04) not in f.calls


# ------------------------------------------------------------------ the generator
def test_the_plugin_is_found_by_name_and_checks_its_task():
    assert registry.get("generator", "moveit") is MoveItGenerator
    with pytest.raises(DatagenError, match="expert for"):
        MoveItGenerator(type("S", (), {"task": "stack_cups"})(), {})
    MoveItGenerator(type("S", (), {"task": "open_drawer"})(), {})


def test_drive_prepares_once_runs_the_expert_and_turns_a_crash_into_notes(monkeypatch):
    built = []

    class Expert:
        def __init__(self, s): built.append(s)
        def run(self): return "ran" if len(built) == 1 and not Expert.crash else self.boom()
        crash = False
        def boom(self): raise RuntimeError("no plan")

    monkeypatch.setitem(experts.EXPERTS, "open_drawer", Expert)
    monkeypatch.setattr(setup, "build", lambda client, instance, task, robot, repo: ("setup", task, robot))
    g = MoveItGenerator(None, {})
    assert g.drive(None, "i", "open_drawer", "r") == {"expert": EXPERT, "notes": "ran"}      # prepare on demand
    g.prepare(None, "i", "open_drawer", "r")                                                  # an explicit prepare rebuilds
    assert built == [("setup", "open_drawer", "r")] * 2
    Expert.crash = True
    out = g.drive(None, "i", "open_drawer", "r")
    assert out["expert"] == EXPERT and out["notes"] == "error: RuntimeError: no plan" and len(built) == 2


# ------------------------------------------------------------------ resume.py: the entry from the current state
PANDA, SO101 = setup.CASES["mobile_panda"], setup.CASES["mobile_so101_xl"]


@pytest.mark.parametrize("case, situation, entry", [
    (PANDA, R.Situation(0.80, 0.04, 0.0), R.APPROACH),                 # far, open hand, closed door
    (PANDA, R.Situation(0.80, 0.04, 0.4, "approach"), R.APPROACH),     # far with the door ajar: approach the moved handle
    (PANDA, R.Situation(0.10, 0.04, 0.0), R.GRASP),                    # the pre-grasp
    (PANDA, R.Situation(0.14, 0.04, 0.0), R.GRASP),
    (PANDA, R.Situation(0.16, 0.04, 0.0), R.APPROACH),
    (PANDA, R.Situation(0.01, 0.04, 0.0), R.GRASP),                    # at the bar, hand still open
    (PANDA, R.Situation(0.01, 0.006, 0.0, "grasp"), R.OPEN),           # closed on the bar
    (PANDA, R.Situation(0.01, 0.006, 0.5, "open"), R.OPEN),
    (PANDA, R.Situation(0.08, 0.006, 0.5, "open"), R.OPEN),            # the evaluator is in `open` and the hand is closed
    (PANDA, R.Situation(0.08, 0.006, 0.5, "approach"), R.GRASP),       # closed but away from the bar, phase not open
    (PANDA, R.Situation(0.50, 0.006, 0.0), R.APPROACH),                # a closed hand far away
    (SO101, R.Situation(0.01, -0.2, 0.0, "open"), R.OPEN),
    (SO101, R.Situation(0.01, -0.02, 0.0), R.GRASP),                   # barely moved from open: not closed
    (SO101, R.Situation(0.10, 0.0, 0.0), R.GRASP),
])
def test_the_entry_follows_the_hand_and_the_distance_to_the_handle(case, situation, entry):
    assert R.entry_for(situation, case) == entry


def test_a_door_or_drawer_moved_beyond_the_tolerance_is_ajar():
    assert not R.is_ajar(R.Situation(0.5, 0.04, 0.01)) and R.is_ajar(R.Situation(0.5, 0.04, 0.05))


def test_max_delta_action_is_the_largest_step_at_the_policy_rate_in_the_window():
    trace = [(t / 100, [0.0, 0.0]) for t in range(0, 40)] + [(t / 100, [0.3, 0.0]) for t in range(40, 200)]
    assert R.max_delta_action(trace, 0.0) == pytest.approx(0.3)           # one jump between two 30 Hz steps
    assert R.max_delta_action(trace, 0.5) == 0.0                          # settled after it
    assert R.max_delta_action(trace, 0.0, window_s=0.2) == 0.0            # the jump is outside the window
    assert R.max_delta_action([], 0.0) is None and R.max_delta_action([(5.0, [0.0])], 0.0) is None


# ------------------------------------------------------------------ resuming the experts on the fake robot
def resumed(task, robot, situation, **kw):
    S, f = fake_robot(task, robot, **kw)
    f.articulation = situation.articulation or None
    f.hand = situation.gripper
    cls = experts.EXPERTS[task]
    e = cls(S)
    notes = e.resume(situation)
    return S, f, e, notes


@needs_assets
def test_resuming_at_the_pre_grasp_holds_the_pose_and_reaches_without_folding_or_driving():
    S, f, e, notes = resumed("open_fridge_door", "mobile_panda", R.Situation(0.10, 0.04, 0.0), arm_error=0.3)
    assert f.calls[0] == ("arm", experts.SETTLE_S) and ("arm", 3.0) not in f.calls                 # held, never folded
    assert not [c for c in f.calls if c[0] == "drive_to"]
    assert plans(f) == ["ompl_rrtc", "pilz_lin", "pilz_lin", "pilz_circ"] and ("hand", S.case["hand_squeeze"]) in f.calls
    assert f.calls[1] == ("hand", 0.04) and ("scene", S.handle.name) in f.calls                      # the hand as found; the bar skipped
    assert '"entry": "grasp"' in notes and '"orbit_done"' in notes and '"first_second"' in notes


@needs_assets
def test_resuming_far_from_the_handle_folds_if_needed_drives_and_does_the_whole_task():
    S, f, e, notes = resumed("open_fridge_door", "mobile_panda", R.Situation(0.9, 0.04, 0.0), arm_error=0.3)
    assert [c[0] for c in f.calls[:3]] == ["arm", "hand", "arm"] and f.calls[2] == ("arm", 3.0)       # hold, hand, fold
    assert f.calls[3][0] == "drive_to" and plans(f) == ["ompl_rrtc", "pilz_lin", "pilz_lin", "pilz_circ"]


@needs_assets
def test_resuming_with_the_bar_held_opens_from_the_doors_angle_without_planning_a_reach():
    held = R.Situation(0.0, 0.006, 0.3, "grasp")
    S, f, e, notes = resumed("open_fridge_door", "mobile_panda", held)
    assert plans(f) == ["pilz_circ"] and ("hand", S.case["hand_squeeze"]) in f.calls                # keeps squeezing, no reach
    assert "open_hand" not in notes and "squeeze" not in notes and not [c for c in f.calls if c[0] == "drive_to"]
    seg = f.calls[-1]
    assert seg[0] == "segment" and seg[2] == pytest.approx(experts.OpenFridgeDoor.ARC_TOTAL - experts.OpenFridgeDoor.ARC_ARM)
    S, f, e, notes = resumed("open_fridge_door", "mobile_panda", R.Situation(0.0, 0.006, 0.5, "open"))   # past the arc already
    assert plans(f) == [] and f.calls[-1][2] == pytest.approx(experts.OpenFridgeDoor.ARC_TOTAL - 0.5)


@needs_assets
def test_resuming_with_the_door_ajar_and_the_hand_off_the_bar_reaches_for_the_moved_handle():
    S, f, e, notes = resumed("open_fridge_door", "mobile_panda", R.Situation(0.9, 0.04, 0.3))
    closed = G.stand_off(S.T_handle, experts.OpenFridgeDoor.DIST, experts.OpenFridgeDoor.SIDE, experts.OpenFridgeDoor.YAW_OFF)
    x, y, yaw = G.stand_off(e.handle_now(), experts.OpenFridgeDoor.DIST, experts.OpenFridgeDoor.SIDE,
                            experts.OpenFridgeDoor.YAW_OFF)
    drive = next(c for c in f.calls if c[0] == "drive_to")
    assert drive[1] == pytest.approx((x, y), abs=1e-3) and drive[1] != pytest.approx(closed[:2], abs=0.05)
    assert e.T_cur is not e.T_h and '"ajar": true' in notes


@needs_assets
def test_a_closed_hand_off_the_bar_is_opened_before_the_reach():
    S, f, e, notes = resumed("open_fridge_door", "mobile_panda", R.Situation(0.1, 0.006, 0.0, "approach"))
    assert ("hand", S.case["hand_open"]) in f.calls and f.calls.index(("hand", S.case["hand_open"])) < f.calls.index(("ik",))
    assert f.calls[1] == ("hand", S.case["hand_squeeze"])                                           # held closed for the first tick


@needs_assets
def test_the_drawer_resumes_its_pull_from_the_slide_it_has():
    S, f, e, notes = resumed("open_drawer", "mobile_so101_xl", R.Situation(0.0, -0.2, 0.12, "open"), bias=[0.0, 0.0, 0.005])
    assert plans(f) == [] and f.calls[-1][2] == pytest.approx(experts.OpenDrawer.PULL - 0.12)
    S, f, e, notes = resumed("open_drawer", "mobile_so101_xl", R.Situation(0.0, -0.2, 0.2, "open"))   # pulled already
    assert not [c for c in f.calls if c[0] == "segment"] and '"pull_done"' in notes


@needs_assets
def test_resume_reads_the_situation_from_the_robot_when_none_is_given():
    S, f = fake_robot("open_fridge_door", "mobile_panda")
    f.tcp = np.append(S.T_handle[:3, 3] + 0.10 * S.T_handle[:3, 0], 1.0)                              # the pre-grasp
    e = experts.OpenFridgeDoor(S)
    s = e.observe()
    assert s.tcp_to_handle == pytest.approx(0.10 - 0.006, abs=1e-3) and s.gripper == 0.04 and s.articulation == 0.0
    assert R.entry_for(s, S.case) == R.GRASP


@needs_assets
def test_each_resume_logs_its_own_notes():
    S, f = fake_robot("open_fridge_door", "mobile_panda")
    e = experts.OpenFridgeDoor(S)
    first = e.resume(R.Situation(0.10, 0.04, 0.0))
    second = e.resume(R.Situation(0.10, 0.04, 0.0))
    assert first == second and second.count('"resume"') == 1


@needs_assets
def test_run_is_the_scenes_start_and_does_not_resume():
    S, f = fake_robot("open_fridge_door", "mobile_panda")
    notes = experts.OpenFridgeDoor(S).run()
    assert '"resume"' not in notes and f.calls[0][0] == "drive_to"           # no hold, no entry choice


def test_the_generator_resumes_only_when_asked(monkeypatch):
    class Expert:
        def __init__(self, s): ...
        def run(self): return "ran"
        def resume(self): return "resumed"

    monkeypatch.setitem(experts.EXPERTS, "open_drawer", Expert)
    monkeypatch.setattr(setup, "build", lambda client, instance, task, robot, repo: None)
    assert MoveItGenerator(None, {}).drive(None, "i", "open_drawer", "r")["notes"] == "ran"
    assert MoveItGenerator(None, {"resume": True}).drive(None, "i", "open_drawer", "r")["notes"] == "resumed"
