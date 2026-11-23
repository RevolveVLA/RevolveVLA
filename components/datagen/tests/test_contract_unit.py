"""contract.py: the manifest check on a golden output (tests/data/contract_golden), the bag check on a synthetic bag
(the frozen topics, JPEG cameras at 30 Hz, /tf_static, the 30 Hz action series)."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from datagen import contract

GOLDEN = Path(__file__).parent / "data" / "contract_golden"
ARM, HAND, CMD, CAM = "/arm_controller/controller_state", "/hand_controller/controller_state", "/omni/cmd_vel", "/cam/image/compressed"
EXPECTED = ["/clock", "/tf_static", ARM, HAND, CMD, "/w/model/r/link/l/sensor/contact/contact", CAM]


@pytest.fixture
def out(tmp_path):
    shutil.copytree(GOLDEN, tmp_path / "o")
    return tmp_path / "o"


def edit(out, name, f):
    p = out / name
    p.write_text(json.dumps(f(json.loads(p.read_text()))))


def test_the_golden_manifest_is_valid(out):
    assert contract.check_manifest(out) == []


@pytest.mark.parametrize("name, f, word", [
    ("manifest.json", lambda m: m.pop("created") and m, "created missing"),
    ("manifest.json", lambda m: m["episodes"][0].pop("datagen") and m, "datagen missing"),
    ("manifest.json", lambda m: m["episodes"][0].update(result="scene_0/nope.json") or m, "result scene_0/nope.json"),
    ("manifest.json", lambda m: m["episodes"][0].update(bag="scene_0") or m, "not a bag directory"),
    ("manifest.json", lambda m: m["episodes"][0].update(passed=False) or m, "passed differs"),
    ("manifest.json", lambda m: m["counts"].update(attempted=2) or m, "counts.attempted"),
    ("manifest.json", lambda m: m["episodes"][0].update(robot="other") or m, "robot 'other'"),
])
def test_a_broken_manifest_is_reported(out, name, f, word):
    edit(out, name, f)
    assert any(word in p for p in contract.check_manifest(out))


def test_a_missing_manifest_and_a_bad_instance_are_reported(out):
    (out / "scene_0" / "scene.yaml").write_text("name: kitchen_partnet\nmodels: {x: {model: nope}}\nrobot_start: 3\n")
    assert any(p.startswith("instance scene_0/scene.yaml") for p in contract.check_manifest(out))
    (out / "manifest.json").unlink()
    assert contract.check_manifest(out)[0].startswith("manifest.json")


# ------------------------------------------------------------------ bags
def state(t, q):
    return NS(header=NS(stamp=NS(sec=int(t), nanosec=int(t % 1 * 1e9))), reference=NS(positions=q))


def twist(t, vx):
    return NS(header=NS(stamp=NS(sec=int(t), nanosec=int(t % 1 * 1e9))), twist=NS(linear=NS(x=vx, y=0.0, z=0.0), angular=NS(z=0.0)))


def bag(cam_hz=30.0, drop=(), fmt="rgb8; jpeg compressed bgr8", span=2.0):
    ts = [i * 0.001 for i in range(int(span * 1000))]
    times = {t: list(ts) for t in EXPECTED if "contact" not in t}
    times[CAM] = [i / cam_hz for i in range(int(span * cam_hz))]
    msgs = {ARM: [state(t, [t, 0.0]) for t in ts], HAND: [state(t, [0.04]) for t in ts],
            CMD: [twist(i / 30, 0.2) for i in range(int(span * 30))], CAM: [NS(format=fmt)] * 5}
    types = {t: "T" for t in times if t not in drop}
    return types, times, msgs


def check(images=True, **kw):
    return contract.check_bag(*bag(**kw), EXPECTED, images)


def test_a_good_bag_passes_and_reports_the_action_series():
    bad, rep = check()
    assert bad == [] and rep["camera_hz"] == {CAM: 30.0}
    assert rep["action"]["hz"] == pytest.approx(30.0, abs=0.1) and rep["action"]["dims"]["cmd_vel"][1] == 3


def test_an_on_demand_topic_may_be_absent_but_others_may_not():
    assert check(drop=[CMD])[0] == []
    assert check(drop=["/clock"])[0] == ["topic /clock missing"]
    assert "topic /tf_static missing" in check(drop=["/tf_static"])[0]


def test_cameras_must_be_jpeg_at_about_30_hz():
    assert any("15.0 Hz" in p for p in check(cam_hz=15.0)[0])
    assert any("not JPEG" in p for p in check(fmt="rgb8")[0])
    assert check(images=False, cam_hz=15.0, fmt="rgb8")[0] == []


def test_no_controller_state_or_a_too_short_one_is_a_problem():
    types, times, msgs = bag()
    assert contract.check_bag(types, times, {**msgs, ARM: []}, EXPECTED, True)[0][-1].startswith("no controller_state")
    assert "no action series" in check(span=0.02)[0][-1]


def test_action_series_holds_the_last_reference_and_zeroes_a_stale_twist():
    st = {"j": [state(0.0, [1.0]), state(1.0, [2.0]), state(2.0, [3.0])]}
    grid, s = contract.action_series(st, [twist(0.0, 0.5)], 0.0, 2.0)
    assert len(grid) == 60 and s["j"][0, 0] == 1.0 and s["j"][-1, 0] == 2.0
    assert s["cmd_vel"][0, 0] == 0.5 and s["cmd_vel"][-1, 0] == 0.0 and np.isfinite(s["cmd_vel"]).all()
