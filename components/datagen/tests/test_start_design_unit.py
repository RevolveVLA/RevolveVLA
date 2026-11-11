"""start_design's `mode: reach` and `mode: drive` starts blocks (host only, no docker): parsing, the per-seed sample
(deterministic, in range), the attempts (attempt 0 = the sample, the rest small jitters), the turn about the first
joint axis, the drive block's far start box, and the train / held-out seeds of the experiment that uses it (disjoint,
distinct samples)."""
import math
from pathlib import Path

import pytest

import specs_bridge
import start_design as S
from revolve_common import specs

REPO = Path(__file__).resolve().parents[3]
BLOCK = dict(mode="reach", grasp_m=[-0.10, 0.10], d_m=[0.68, 0.76], yaw_half_deg=8.0, jitter_m=0.01, jitter_deg=1.0)
DRIVE = dict(BLOCK, mode="drive", back_m=[1.0, 1.6], side_m=[-0.3, 0.3], heading_half_deg=20.0, slowdown=2.0)


def test_parse():
    spec = S.parse_starts(BLOCK)
    assert isinstance(spec, S.ReachSpec) and spec.window_half_deg == 9.0
    assert S.yaw_window_rad(spec) == pytest.approx((math.pi - math.radians(9), math.pi + math.radians(9)))
    for bad in (dict(BLOCK, mode="box"), dict(BLOCK, d_m=[0.8, 0.7]), dict(BLOCK, d_m=[0.0, 0.7]),
                dict(BLOCK, yaw_half_deg=45), dict(BLOCK, extra=1), {k: v for k, v in BLOCK.items() if k != "jitter_m"}):
        with pytest.raises(ValueError):
            S.parse_starts(bad)
    # a clustered block (no mode) parses as before
    assert isinstance(S.parse_starts(specs_bridge.generator()["tasks"]["open_drawer"]["starts"]), S.StartsSpec)


def test_xl_block_is_data():
    blocks = specs_bridge.round_blocks("open_drawer", "mobile_so101_xl")
    assert S.parse_starts(blocks["starts"]) == S.parse_starts(DRIVE)
    assert "starts" not in specs_bridge.round_blocks("open_drawer", "mobile_so101")


def test_parse_drive():
    spec = S.parse_starts(DRIVE)
    assert isinstance(spec, S.DriveSpec) and isinstance(spec, S.ReachSpec)
    assert (spec.back_m, spec.side_m, spec.heading_half_deg, spec.slowdown) == ((1.0, 1.6), (-0.3, 0.3), 20.0, 2.0)
    assert spec.window_half_deg == 21.0          # the start heading and its jitter bound the base yaw
    for bad in (dict(DRIVE, back_m=[0.0, 1.6]), dict(DRIVE, back_m=[1.6, 1.0]), dict(DRIVE, side_m=[0.3]),
                dict(DRIVE, heading_half_deg=60), dict(DRIVE, slowdown=0.5), {k: v for k, v in DRIVE.items() if k != "back_m"},
                dict(DRIVE, mode="fly")):
        with pytest.raises(ValueError):
            S.parse_starts(bad)


def test_optional_bearing_and_sequence():
    """bearing_deg and sequence are optional (defaults: the robot's reach bearing, simultaneous) and checked."""
    spec = S.parse_starts(DRIVE)
    assert spec.bearing_deg is None and spec.sequence == "simultaneous"
    spec = S.parse_starts(dict(DRIVE, bearing_deg=40.0, sequence="drive_then_arm"))
    assert spec.bearing_deg == 40.0 and spec.sequence == "drive_then_arm"
    assert S.parse_starts(dict(BLOCK, bearing_deg=-20)).bearing_deg == -20.0
    for bad in (dict(DRIVE, sequence="arm_first"), dict(DRIVE, bearing_deg=120), dict(BLOCK, sequence="drive_then_arm")):
        with pytest.raises(ValueError):
            S.parse_starts(bad)


def test_variant_blocks_are_opt_in(monkeypatch):
    """A datagen variant (generator.yaml tasks.<task>.variants) applies only when named; unknown ones fail."""
    monkeypatch.delenv(specs.VARIANT_ENV, raising=False)
    default = specs_bridge.round_blocks("open_fridge_door", "mobile_panda")
    assert isinstance(S.parse_starts(default["starts"]), S.StartsSpec)
    seq = S.parse_starts(specs_bridge.round_blocks("open_fridge_door", "mobile_panda", "drive_seq")["starts"])
    assert isinstance(seq, S.DriveSpec) and seq.sequence == "drive_then_arm" and seq.grasp_m == (0.965, 1.085)
    monkeypatch.setenv(specs.VARIANT_ENV, "drive_seq")
    assert specs_bridge.round_blocks("open_fridge_door", "mobile_panda")["starts"]["sequence"] == "drive_then_arm"
    assert specs_bridge.round_blocks("open_fridge_door", "mobile_panda")["timing"] == default["timing"]
    for task, robot in (("open_drawer", "mobile_panda"), ("open_fridge_door", "mobile_so101_xl")):
        with pytest.raises(ValueError):
            specs_bridge.round_blocks(task, robot)


def test_drive_sample_box_and_attempts():
    spec = S.parse_starts(DRIVE)
    h = math.radians(20.0)
    got = []
    for seed in range(300):
        d = S.drive_sample(seed, spec)
        assert d == S.drive_sample(seed, spec)
        assert 1.0 <= d.back <= 1.6 and -0.3 <= d.side <= 0.3 and -h <= d.heading <= h
        # the reach part (grasp, d, turn) is the reach block's sample of the seed
        assert S.reach_sample(seed, spec) == S.reach_sample(seed, S.parse_starts(BLOCK))
        att = S.drive_attempts(seed, 5, spec)
        assert att[0] == d and all(abs(a.back - d.back) <= 0.01 + 1e-12 and abs(a.side - d.side) <= 0.01 + 1e-12
                                   and abs(a.heading - d.heading) <= math.radians(1) + 1e-12 for a in att[1:])
        got.append(d)
    # the box is covered to its edges
    assert min(x.back for x in got) < 1.02 and max(x.back for x in got) > 1.58
    assert min(x.side for x in got) < -0.28 and max(x.side for x in got) > 0.28
    assert min(x.heading for x in got) < -0.95 * h and max(x.heading for x in got) > 0.95 * h


def test_drive_start_is_behind_the_reach_base():
    reach = (1.40, 0.05, math.pi + 0.1)          # facing the cabinet (-x), turned 0.1 rad
    st = S.drive_start(reach, S.DriveSample(back=1.2, side=0.25, heading=math.radians(-15)))
    # facing -x: back = +x, left of the facing = -y; the heading from the facing, not from the reach turn
    assert st == pytest.approx((2.60, -0.20, math.pi - math.radians(15)))


def test_same_seed_same_sample():
    spec = S.parse_starts(BLOCK)
    for seed in (0, 7, 104):
        assert S.reach_sample(seed, spec) == S.reach_sample(seed, spec)
        assert S.reach_attempts(seed, 50, spec) == S.reach_attempts(seed, 50, spec)
    assert S.reach_sample(0, spec) != S.reach_sample(1, spec)


def test_samples_in_range_and_attempts_jitter():
    spec = S.parse_starts(BLOCK)
    h = math.radians(8.0)
    for seed in range(300):
        s = S.reach_sample(seed, spec)
        assert -0.10 <= s.grasp <= 0.10 and 0.68 <= s.d <= 0.76 and -h <= s.dyaw <= h
        att = S.reach_attempts(seed, 50, spec)
        assert len(att) == 50 and att[0] == (s.d, s.dyaw)
        for d, dyaw in att[1:]:
            assert abs(d - s.d) <= 0.01 + 1e-12 and abs(dyaw - s.dyaw) <= math.radians(1.0) + 1e-12


def test_turn_about_keeps_the_pivot():
    base, pivot = (1.0, 0.2, math.pi), (0.23, 0.0)

    def at(b):
        c, s = math.cos(b[2]), math.sin(b[2])
        return b[0] + c * pivot[0] - s * pivot[1], b[1] + s * pivot[0] + c * pivot[1]
    turned = S.turn_about(base, pivot, math.radians(8))
    assert at(turned) == pytest.approx(at(base)) and turned[2] == pytest.approx(math.pi + math.radians(8))
    starts = S.reach_starts(3, 5, S.parse_starts(BLOCK), lambda d: (0.39 + 0.23 + d, 0.0, math.pi), pivot)
    assert [st.grasp_id for st in starts] == [0] * 5 and starts[0].dx == S.reach_sample(3, S.parse_starts(BLOCK)).d


def test_train_and_heldout_seeds_disjoint():
    e = specs.load(REPO / "specs" / "experiments" / "drawer_so101_xl_f1.yaml")
    seeds = e.experiment.seeds
    train, held = set(seeds.train.seeds()), set(seeds.heldout.seeds())
    assert train == set(range(50)) and held == set(range(100, 110)) and not train & held
    spec = S.parse_starts(BLOCK)
    samples = {S.reach_sample(s, spec) for s in train | held}
    assert len(samples) == 60            # every seed its own (start, grasp)
