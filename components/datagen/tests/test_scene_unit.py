"""scene/ (sdf_boxes, footprint, instance), the robot_start randomiser, the scene_instance instancer and the
collision_free validator: synthetic SDF fixtures (no licence assets), the repo's own specs for names and poses."""
import math

import numpy as np
import pytest
import yaml
from revolve_common import specs

from datagen import cli
from datagen.core import interfaces as I, pipeline, spec as S, stats
from datagen.core.registry import register
from datagen.plugins.randomisers import listing, robot_start as RS
from datagen.plugins.validators import collision_free as CF
from datagen.scene import base, footprint as FP, sdf_boxes as SB
from datagen.scene.instance import Variation

RAW = {"task": "open_fridge_door", "robot": "mobile_panda", "out": "unused", "generator": {"plugin": "scene_gen"},
       "randomisers": {"robot_start": {"x_m": [-1.85, -1.65], "y_m": [1.0, 1.2], "yaw_rad": [1.47, 1.67],
                                       "arm": {"choices": ["folded", "ready"]}}},
       "splits": {"train": {"target": 3, "pool": {"initial": [0, 3], "replacement": [4, 9]}},
                  "heldout": {"target": 2, "pool": {"initial": [20, 24]}, "record": False,
                              "min_distance": {"from": "train", "value": 0.2, "realised": True}}}}
BASE = base.read_scene(S.parse(RAW))["robot_start"]       # kitchen_partnet: base [-1.75, 1.1, pi/2], arm folded


@register("generator", "scene_gen")
class SceneGen(I.Generator):
    """Accepts everything and reports nothing: the pipeline fills `realised` from the scene instance."""

    def __init__(self, spec, params):
        pass

    def generate(self, config, out):
        return I.Result()


def spec_of(**edits):
    return S.parse(RAW | edits)


def config(seed=0, **values):
    return I.Config("train", seed, (), {f"robot_start.{k}": v for k, v in values.items()})


# ------------------------------------------------------------------ sdf_boxes
SDF = """<sdf version="1.9"><model name="m">
  <link name="a"><pose>0 0 1 0 0 {link_yaw}</pose>
    <collision name="box"><pose>1 0 0 0 0 0</pose><geometry><box><size>0.2 0.4 0.6</size></box></geometry></collision>
    <collision name="cyl"><geometry><cylinder><radius>0.1</radius><length>0.5</length></cylinder></geometry></collision>
    <collision name="ball"><geometry><sphere><radius>0.05</radius></sphere></geometry></collision>
    <collision name="ground"><geometry><plane><normal>0 0 1</normal></plane></geometry></collision>
    <collision name="mesh"><geometry><mesh><uri>model://m/hull.obj</uri><scale>2 1 1</scale></mesh></geometry></collision>
  </link>
  <joint name="j" type="revolute"><parent>world</parent><child>a</child></joint>
</model></sdf>"""
OBJ = "v 0 0 0\nv 1 0 0\nv 0 2 0\nv 0 0 3\nf 1 2 3\n"


def make_repo(tmp_path, sdf=SDF.format(link_yaw="1.5707963267948966"), obj=OBJ):
    d = tmp_path / "assets" / "m"
    d.mkdir(parents=True)
    (d / "model.sdf").write_text(sdf)
    if obj is not None:
        (d / "hull.obj").write_text(obj)
    return tmp_path


def scene_of(**models):
    return specs.Scene.model_validate({"name": "s", "assets": ["assets"], "robot_start": {"base": [0, 0, 0], "arm": "folded"},
                                       "models": models or {"thing": {"model": "m", "pose": [2, 3, 0.5, math.pi]}}})


def test_collisions_become_world_boxes(tmp_path):
    boxes = {b.name: b for b in SB.scene_boxes(scene_of(), make_repo(tmp_path))}
    assert set(boxes) == {"thing::a::box", "thing::a::cyl", "thing::a::ball", "thing::a::mesh"}      # the plane is skipped
    box = boxes["thing::a::box"]
    # model (2,3,0.5, yaw pi) @ link (0,0,1, yaw pi/2) @ collision (1,0,0): x = 2 + R(3pi/2)(1,0) = (2, 2)
    assert box.size == (0.2, 0.4, 0.6) and np.allclose(box.pose[:3, 3], [2, 2, 1.5])
    assert np.allclose(box.pose[:3, :3] @ [1, 0, 0], [0, -1, 0], atol=1e-9)              # yaw 3pi/2
    assert boxes["thing::a::cyl"].size == pytest.approx((0.2, 0.2, 0.5))
    assert boxes["thing::a::ball"].size == pytest.approx((0.1,) * 3)
    mesh = boxes["thing::a::mesh"]                                                      # OBJ aabb x scale 2 1 1
    assert mesh.size == pytest.approx((2, 2, 3)) and np.allclose(mesh.pose[:3, 3], [3, 2, 3.0])


def test_box_corners_cover_the_box():
    b = SB.Box("b", SB.tf((1, 2, 3), (0, 0, math.pi / 2)), (2.0, 1.0, 0.5))
    c = b.corners()
    assert c.shape == (8, 3) and np.allclose(c.min(0), [0.5, 1, 2.75]) and np.allclose(c.max(0), [1.5, 3, 3.25])


@pytest.mark.parametrize("sdf, obj, msg", [
    (SDF.format(link_yaw=0), None, "hull.obj: in none of"),
    (SDF.format(link_yaw=0), "", "no vertices"),
    (SDF.replace("<plane><normal>0 0 1</normal></plane>", "<heightmap/>").format(link_yaw=0), OBJ, "unsupported geometry"),
    ("<sdf><model name='m'><link name='a'><pose relative_to='x'>0 0 0 0 0 0</pose></link></model></sdf>", OBJ, "relative_to"),
])
def test_unreadable_geometry_is_a_spec_error(tmp_path, sdf, obj, msg):
    with pytest.raises(S.SpecError, match=msg):
        SB.scene_boxes(scene_of(), make_repo(tmp_path, sdf, obj))


def test_a_missing_model_is_a_spec_error(tmp_path):
    with pytest.raises(S.SpecError, match="model://nope"):
        SB.scene_boxes(scene_of(thing={"model": "nope"}), make_repo(tmp_path))


# ------------------------------------------------------------------ footprint
def box(x, y, size=(1.0, 1.0, 1.0), z=0.5, yaw=0.0):
    return SB.Box(f"b{x},{y}", SB.tf((x, y, z), (0, 0, yaw)), size)


def test_polygon_follows_the_pose():
    fp = FP.Footprint(x=(-0.25, 0.36), y=(-0.25, 0.25))
    poly = fp.polygon(1.0, 2.0, math.pi / 2)                       # facing +y: front edge at y = 2.36
    assert max(p[1] for p in poly) == pytest.approx(2.36) and min(p[1] for p in poly) == pytest.approx(1.75)
    assert max(p[0] for p in poly) == pytest.approx(1.25) and min(p[0] for p in poly) == pytest.approx(0.75)


def test_hull_drops_inner_points_and_keeps_degenerate_sets():
    assert sorted(FP.hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert FP.hull([(0, 0), (1, 1)]) == [(0, 0), (1, 1)]


@pytest.mark.parametrize("b, gap", [
    ([(2, 0), (3, 0), (3, 1), (2, 1)], 1.0),                       # side by side
    ([(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)], math.hypot(0.5, 0.5)),     # corner to corner
    ([(0.5, 0.5), (2, 0.5), (2, 2), (0.5, 2)], 0.0),               # overlapping
    ([(-1, -1), (2, -1), (2, 2), (-1, 2)], 0.0),                   # containing
    ([(1, 0), (2, 0), (2, 1), (1, 1)], 0.0),                       # touching
])
def test_polygon_distance(b, gap):
    assert FP.polygon_distance([(0, 0), (1, 0), (1, 1), (0, 1)], b) == pytest.approx(gap)


def test_clearance_rejects_inside_the_margin_and_names_the_box():
    fp = FP.Footprint(x=(-0.25, 0.36), y=(-0.25, 0.25))
    wall = box(0, 3.0, size=(4, 0.2, 2), z=1.0)                 # face at y = 2.9
    c = FP.Clearance([wall], fp, 0.05)
    assert c.reject(0, 2.0, math.pi / 2) is None               # front at 2.36: gap 0.54
    assert "b0,3.0" in c.reject(0, 2.55, math.pi / 2)          # front at 2.91: overlap
    near = c.reject(0, 2.52, math.pi / 2)                      # front 2.88: gap 0.02 < 0.05
    assert near.startswith("collision: footprint 0.020 m from b0,3.0 (min 0.05 m)")
    assert c.reject(0, 2.52, -math.pi / 2) is None             # turned away: the rear edge is 0.25 back: gap 0.13 ok


def test_the_arm_overhang_is_part_of_the_footprint():
    wall = box(0, 3.0, size=(4, 0.2, 2), z=1.0)
    panda = FP.Clearance([wall], FP.footprint_of("mobile_panda"), 0.05)
    bare = FP.Clearance([wall], FP.footprint_of("mobile_so101"), 0.05)           # no entry: the bare chassis
    assert FP.footprint_of("mobile_panda").x[1] == 0.36 and FP.footprint_of("mobile_so101_xl").x[1] == 0.51
    assert panda.reject(0, 2.5, math.pi / 2) and bare.reject(0, 2.5, math.pi / 2) is None


def test_only_boxes_at_the_robot_height_count():
    fp = FP.Footprint()
    above = box(0.5, 0, size=(1, 1, 0.2), z=1.5)               # a lintel over the robot
    slab = box(0.5, 0, size=(1, 1, 0.01), z=0.005)             # a rug
    assert FP.Clearance([above, slab], fp, 0.05).hulls == []
    assert len(FP.Clearance([box(0.5, 0)], fp, 0.05).hulls) == 1


def test_a_rotated_box_is_measured_by_its_hull():
    fp = FP.Footprint()
    diamond = box(1.0, 0, size=(0.5, 0.5, 1), yaw=math.pi / 4)  # corner points at x = 1 - 0.354
    c = FP.Clearance([diamond], fp, 1.0)
    assert c.nearest(0, 0, 0)[0] == pytest.approx(1.0 - 0.3536 - 0.25, abs=1e-3)


# ------------------------------------------------------------------ robot_start
def test_limits_are_a_box_around_the_base_start_and_the_robots_poses():
    lim = RS.RobotStart().limits(spec_of())
    assert lim["robot_start.x_m"] == S.Range(-1.75 - 1.5, -1.75 + 1.5)
    assert lim["robot_start.yaw_rad"] == S.Range(math.pi / 2 - math.pi, math.pi / 2 + math.pi)
    assert lim["robot_start.dx_m"] == S.Range(-1.5, 1.5) and lim["robot_start.dyaw_rad"] == S.Range(-math.pi, math.pi)
    assert lim["robot_start.arm"] == S.Choices(("ready", "folded"))
    assert set(lim) == {f"robot_start.{k}" for k in (*RS.ABSOLUTE, *RS.OFFSET, "arm")}


def test_realise_absolute_offset_and_unvaried_dims():
    r = RS.RobotStart()
    ctx = {"base": BASE}
    assert r.realise({"x_m": -1.8, "y_m": 1.15, "yaw_rad": 1.5, "arm": "ready"}, ctx) == {"base": [-1.8, 1.15, 1.5], "arm": "ready"}
    assert r.realise({}, ctx) == {"base": list(BASE["base"][:2]) + [round(BASE["base"][2], 6)], "arm": "folded"}
    out = r.realise({"dx_m": 0.1, "y_m": 2.0, "dyaw_rad": -0.2}, ctx)               # offsets and absolutes mix per axis
    assert out["base"] == [round(-1.75 + 0.1, 6), 2.0, round(math.pi / 2 - 0.2, 6)]


def test_measure_reports_every_dim_form():
    r = RS.RobotStart()
    real = r.realise({"dx_m": 0.1, "dy_m": -0.05, "arm": "ready"}, {"base": BASE})
    m = r.measure(real, BASE)
    assert m["dx_m"] == pytest.approx(0.1) and m["dy_m"] == pytest.approx(-0.05) and m["dyaw_rad"] == 0
    assert m["x_m"] == pytest.approx(-1.65) and m["arm"] == "ready"


@pytest.mark.parametrize("edits, where, msg", [
    ({"x_m": [-1.85, 5.0]}, "randomisers.robot_start.x_m", "outside the hard limits"),
    ({"arm": {"choices": ["folded", "tucked"]}}, "randomisers.robot_start.arm", "outside the hard limits"),
    ({"zz_m": [0, 1]}, "randomisers.robot_start.zz_m", "unknown dim"),
    ({"dx_m": [-0.1, 0.1]}, "randomisers.robot_start", "x_m and dx_m both set"),
    ({"dyaw_rad": [-4.0, 4.0], "yaw_rad": None}, "randomisers.robot_start.dyaw_rad", "outside the hard limits"),
])
def test_invalid_ranges_name_the_yaml_path(edits, where, msg):
    dims = {k: v for k, v in (RAW["randomisers"]["robot_start"] | edits).items() if v is not None}
    spec = spec_of(randomisers={"robot_start": dims})
    with pytest.raises(S.SpecError) as e:
        S.check_limits(spec, RS.RobotStart().limits(spec))
    assert where in str(e.value) and msg in str(e.value)


def test_the_base_scene_defaults_to_the_tasks_and_a_bad_task_is_one_line():
    assert base.scene_name(spec_of()) == "kitchen_partnet" and base.scene_name(spec_of(scene="x")) == "x"
    with pytest.raises(S.SpecError, match="task 'nope'"):
        RS.RobotStart().limits(spec_of(task="nope"))
    with pytest.raises(S.SpecError, match="scene 'nope'"):
        RS.RobotStart().limits(spec_of(scene="nope"))


def test_spec_scene_key_and_hash():
    assert S.parse(RAW).scene is None and S.parse(RAW | {"scene": "kitchen_partnet"}).scene == "kitchen_partnet"
    assert S.parse(RAW).hash != S.parse(RAW | {"scene": "kitchen_partnet"}).hash
    with pytest.raises(S.SpecError, match="<spec>.scene"):
        S.parse(RAW | {"scene": ""})


# ------------------------------------------------------------------ instances
def test_an_instance_is_the_base_scene_with_the_new_start_and_loads(tmp_path):
    spec = spec_of()
    inst = cli.registry.create("instancer", "scene_instance", spec, {})
    res = inst.write(config(7, x_m=-1.7, y_m=1.05, yaw_rad=1.6, arm="ready"), tmp_path / "scene_7")
    path = tmp_path / "scene_7" / "scene.yaml"
    assert res.paths == {"scene": str(path)} and res.realised["robot_start.x_m"] == -1.7
    data, base_data = yaml.safe_load(path.read_text()), base.read_scene(spec)
    assert data["robot_start"] == {"base": [-1.7, 1.05, 1.6], "arm": "ready"}
    assert {k: v for k, v in data.items() if k != "robot_start"} == {k: v for k, v in base_data.items() if k != "robot_start"}
    scene = specs.load(path, assets=False)                          # outside specs/: a scene instance
    assert scene.name == "kitchen_partnet" and scene.robot_start.arm == "ready" and scene.robot_start.base == [-1.7, 1.05, 1.6]
    assert list(path.parent.iterdir()) == [path]


def test_instances_are_deterministic_per_seed():
    var = Variation(spec_of())
    c = config(3, x_m=-1.7, arm="folded")
    assert var.data(c) == Variation(spec_of()).data(c) and var.data(c) != var.data(config(3, x_m=-1.8))


def test_a_spec_without_a_scene_randomiser_has_no_instancer():
    spec = S.parse({k: v for k, v in RAW.items() if k != "randomisers"} | {"randomisers": {"fake_datagen": {"x": [0, 1]}}})
    import fake_datagen                                                  # noqa: F401  (registers the randomiser)
    assert cli.build_plugins(spec, ("sample",)).instancer is None
    assert cli.build_plugins(spec_of(), ("check",)).instancer is None        # `check` writes nothing
    assert cli.build_plugins(spec_of(), ("sample",)).instancer is not None


def test_the_old_randomisers_are_refused_with_the_replacement():
    spec = S.parse(RAW | {"randomisers": {"robot_base": {"x_m": [-0.1, 0.1]}}})
    with pytest.raises(cli.DatagenError, match=r"randomisers.robot_base: removed.*use robot_start"):
        cli.build_plugins(spec, ("sample",))


def test_the_listing_shows_the_new_dims():
    out = listing.render("open_fridge_door", "mobile_panda")
    assert out.splitlines()[0] == "open_fridge_door on mobile_panda: scene kitchen_partnet"
    assert "robot_start.yaw_rad" in out and "{ready, folded}" in out
    with pytest.raises(cli.DatagenError, match="--list-randomisers"):
        listing.render("open_fridge_door", "nope")


# ------------------------------------------------------------------ the pipeline
def make(tmp_path, steps=pipeline.STEPS, dry_run=False, **edits):
    spec = spec_of(**edits)
    return pipeline.Pipeline(spec, cli.build_plugins(spec, steps), tmp_path, dry_run, log=lambda m: None)


def instance_dirs(tmp_path, split):
    return sorted(int(p.name.removeprefix("scene_")) for p in (tmp_path / split).glob("scene_*"))


def test_sample_writes_one_loadable_instance_per_sampled_config(tmp_path):
    make(tmp_path).run(["sample"])
    assert instance_dirs(tmp_path, "train") == list(range(10)) and instance_dirs(tmp_path, "heldout") == list(range(20, 25))
    for seed in (0, 9):
        s = specs.load(tmp_path / "train" / f"scene_{seed}" / "scene.yaml", assets=False)
        assert -1.85 <= s.robot_start.base[0] <= -1.65 and s.robot_start.arm in ("folded", "ready")


def test_resume_keeps_existing_instances_and_dry_run_writes_none(tmp_path):
    make(tmp_path / "dry", dry_run=True).run(["sample"])
    assert not (tmp_path / "dry").exists()
    make(tmp_path).run(["sample"])
    keep = tmp_path / "train" / "scene_0" / "scene.yaml"
    keep.write_text("sentinel")
    (tmp_path / "train" / "scene_1" / "scene.yaml").unlink()
    make(tmp_path).run(["sample"])
    assert keep.read_text() == "sentinel" and (tmp_path / "train" / "scene_1" / "scene.yaml").exists()


def blocked(*xs):
    """Boxes standing where a start of x_m in `xs` (+-0.01) would put the robot."""
    return [SB.Box(f"block{x}", SB.tf((x, 1.1, 0.5)), (0.02, 3.0, 1.0)) for x in xs]


def test_a_colliding_start_is_rejected_and_replaced_from_the_pool(tmp_path, monkeypatch):
    wide = {"robot_start": RAW["randomisers"]["robot_start"] | {"x_m": [-3.2, -0.3]}}
    spec = spec_of(randomisers=wide, validators=[{"plugin": "collision_free"}])
    pl = pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path, log=lambda m: None)
    pl.run(["sample"])
    sampled = {c["seed"]: I.Config.from_dict(c) for c in pl.manifest.split("train")["sampled"]}
    x1 = Variation(spec).realisation(sampled[1])["robot_start"]["base"][0]
    monkeypatch.setattr(CF, "scene_boxes", lambda scene, repo: blocked(x1))
    pl = pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path, log=lambda m: None)
    pl.run(["generate"], ["train"])
    rec = pl.manifest.split("train")
    rejected = {e["seed"]: e for e in rec["rejected"]}
    assert rejected[1]["step"] == "validate" and rejected[1]["reason"].startswith("collision: footprint")
    assert len(rec["used"]) == 3 and not {e["config"]["seed"] for e in rec["used"]} & set(rejected)
    assert (tmp_path / "train" / "scene_1" / "scene.yaml").exists()             # kept, to see why


def test_collision_free_needs_a_robot_start_and_known_footprint_keys():
    with pytest.raises(S.SpecError, match="does not vary robot_start"):
        CF.CollisionFree(S.parse({k: v for k, v in RAW.items() if k != "randomisers"}), {})
    v = CF.CollisionFree(spec_of(), {"footprint": {"x": [-0.1, 0.2], "q": [0, 1]}})
    with pytest.raises(S.SpecError, match="unknown keys"):
        v.footprint()
    assert CF.CollisionFree(spec_of(), {"footprint": {"x": [-0.1, 0.2]}}).footprint().x == (-0.1, 0.2)


def test_the_realised_start_is_the_instances_and_drives_min_distance(tmp_path):
    pl = make(tmp_path)
    pl.run(["sample", "generate"])
    m = pl.manifest.data["splits"]
    train = m["train"]["used"][0]
    sampled = train["config"]["values"]
    assert train["generate"]["realised"]["robot_start.x_m"] == pytest.approx(sampled["robot_start.x_m"], abs=1e-6)
    assert train["generate"]["realised"]["robot_start.arm"] == sampled["robot_start.arm"]
    d = stats.distances(pl.spec, pl.spec.split("heldout"), pl.manifest.data)
    assert d["realised"] is not None and d["realised"][0] >= 0.2 - 1e-6 and not d["unreported"]


# ------------------------------------------------------------------ the real scene (assets are git-ignored: skipped without)
ASSETS = specs.REPO_SPECS.parent / "assets" / "generated_partnet" / "fridge" / "model.sdf"


@pytest.mark.skipif(not ASSETS.is_file(), reason="needs ./revolve assets")
@pytest.mark.parametrize("pose, collides", [
    ((-1.75, 1.1, math.pi / 2), False),         # the base scene's start: 0.23 m from the counter, 0.9 m from the fridge
    ((-1.75, 1.9, math.pi / 2), False),
    ((-1.75, 2.2, math.pi / 2), True),          # into the fridge
    ((-2.1, 0.3, math.pi / 2), True),           # into the counter
    ((-3.2, 1.1, math.pi), True),               # into the west wall
    ((1.5, -0.5, 0.0), True),                   # onto table_01
    ((0.0, 0.0, 0.0), False),
])
def test_the_real_kitchen(pose, collides):
    clearance = CF.CollisionFree(spec_of(validators=[{"plugin": "collision_free"}]), {}).clearance()
    assert (clearance.reject(*pose) is not None) == collides
