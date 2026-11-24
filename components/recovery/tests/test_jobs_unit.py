"""core/jobs.py: the job's scene instance, the Warm client, run_job on the fake sim (relaunch and warm), resume."""
import fake_recovery as F
import pytest
import yaml
from datagen.core import episodes
from datagen.core.spec import EpisodeSpec
from recovery.core import jobs as J
from recovery.core.errors import JobError
from recovery.core.failures import Failure
from recovery.core.restart import Restart
from recovery.core.robot import Arm
from revolve_common import specs
from revolve_common.sim_client import WarmResetMissed
from revolve_common.sim_fake import FakeSession
from revolve_common.warm_reset import ResetTarget

ARM = Arm("panda_arm_controller", tuple(F.ARM))
RESTART = Restart(10.0, (-1.5, 1.7, 1.5), tuple(0.1 * i for i in range(7)), 0.0)


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "eval" / "scene.yaml"
    path.parent.mkdir()
    base = yaml.safe_load((specs.REPO_SPECS / "scenes" / "kitchen_partnet" / "scene.yaml").read_text())
    path.write_text(yaml.safe_dump(base))
    return path


def failure(scene, restart=RESTART):
    return Failure("mined_100_r0_t0", "mined", "100_r0", 0, str(scene), "grasp", "verdict", "grasp:timeout_s", 11.0, 6.0, "bag",
                   "result", restart)


def test_make_writes_the_scene_at_the_restart(tmp_path, scene):
    job = J.make(3, failure(scene), tmp_path / "out", "mobile_panda", ARM)
    assert job.dir == tmp_path / "out" / "grasp" / "mined_100_r0_t0" and job.index == 3 and job.scene == job.dir / "scene.yaml"
    written = yaml.safe_load(job.scene.read_text())
    assert written["robot_start"] == {"base": [-1.5, 1.7, 1.5], "arm": [round(0.1 * i, 6) for i in range(7)]}
    assert written["models"] == yaml.safe_load(scene.read_text())["models"] and written["name"] == "kitchen_partnet"
    assert job.target == ResetTarget("mobile_panda", (-1.5, 1.7, 1.5), RESTART.arm, ARM.joints, "panda_arm_controller")
    assert specs.load(job.scene, assets=False).robot_start.base == [-1.5, 1.7, 1.5]          # a scene the specs accept


def test_make_with_an_unreadable_scene(tmp_path, scene):
    scene.unlink()
    with pytest.raises(JobError, match="cannot read the scene"):
        J.make(0, failure(scene), tmp_path / "out", "mobile_panda", ARM)


def test_job_json_round_trip(tmp_path, scene):
    job = J.make(1, failure(scene), tmp_path / "out", "mobile_panda", ARM)
    assert J.Job.from_json(job.to_json()) == job


def session(tmp_path, scene, **kw):
    s = FakeSession(scene, "mobile_panda", "open_fridge_door", tmp_path / "sim", "sim-0", 5, **kw)
    s.start()
    return s


def test_relaunch_job_runs_the_generator_and_logs_the_sim_folder(tmp_path, scene):
    job = J.make(0, failure(scene), tmp_path / "out", "mobile_panda", ARM)
    s = session(tmp_path, job.scene)
    F.CALLS.clear()
    log = J.run_job(s.connect(), F.FakeGenerator(None, {}), job, "open_fridge_door", "mobile_panda", EpisodeSpec(2, 1), s.out)
    assert [c[1] for c in F.CALLS] == [job.scene]
    assert episodes.read(job.dir) == log and log["seed"] == 0
    (t,) = log["tries"]
    assert t["passed"] and t["sim"] == str(s.out) and t["expert"] == "fake_v1" and (s.out / "episode_0.json").is_file()
    assert J.resets(job.dir) == [] and s.client_state["resets"] == 0                # a fresh sim: no reset at all


def test_warm_job_resets_to_the_restart_before_every_try_the_first_included(tmp_path, scene):
    job = J.make(0, failure(scene), tmp_path / "out", "mobile_panda", ARM)
    s = session(tmp_path, job.scene, passes=lambda n: n >= 1)
    log = J.run_job(s.connect(), F.FakeGenerator(None, {}), job, "open_fridge_door", "mobile_panda", EpisodeSpec(3, 1), s.out, True)
    assert [t["passed"] for t in log["tries"]] == [False, True] and s.warm_targets == [job.target, job.target]
    assert [(r["attempt"], r["mode"], r["ok"]) for r in J.resets(job.dir)] == [(0, "warm", True), (1, "warm", True)]


def test_a_missed_warm_reset_stops_the_job_with_its_report(tmp_path, scene):
    job = J.make(0, failure(scene), tmp_path / "out", "mobile_panda", ARM)
    s = session(tmp_path, job.scene, reaches=lambda target: False)
    with pytest.raises(WarmResetMissed, match="missed the target"):
        J.run_job(s.connect(), F.FakeGenerator(None, {}), job, "open_fridge_door", "mobile_panda", EpisodeSpec(2, 1), s.out, True)
    assert episodes.read(job.dir)["tries"] == [] and J.resets(job.dir)[0]["ok"] is False and not list(s.out.glob("episode_*.json"))


def test_a_finished_job_is_not_run_again(tmp_path, scene):
    job = J.make(0, failure(scene), tmp_path / "out", "mobile_panda", ARM)
    s = session(tmp_path, job.scene)
    J.run_job(s.connect(), F.FakeGenerator(None, {}), job, "open_fridge_door", "mobile_panda", EpisodeSpec(2, 1), s.out)
    F.CALLS.clear()
    J.run_job(s.connect(), F.FakeGenerator(None, {}), job, "open_fridge_door", "mobile_panda", EpisodeSpec(2, 1), s.out)
    assert F.CALLS == [] and len(episodes.read(job.dir)["tries"]) == 1
