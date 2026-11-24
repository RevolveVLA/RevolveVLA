"""The fake eval run itself: it reads back through the real BagSource and supervision."""
import fake_recovery as F
import pytest
from revolve_common import specs
from revolve_common.supervision import BagSource, build


def test_fake_run_reads_back(tmp_path):
    run = F.write_eval_run(tmp_path, [F.Try(failed="grasp")])
    task = specs.resolve(specs.REPO_SPECS, "open_fridge_door", assets=False)
    rec = F.json.loads((run / "results.json").read_text())["sets"]["mined"]["rollouts"]["100_r0"]["tries"][0]["record"]
    src = BagSource(F.read_messages, rec["bag"], F.ROBOT, F.json.loads(F.Path(rec["result_file"]).read_text()))
    firing = build([{"plugin": "verdict"}], task, arm_joints=F.ARM).scan(src.ticks())
    assert (firing.phase, firing.t) == ("grasp", 11.0)
    now = src.state_at(10.0)
    assert now.objects == {"fridge::top_door_hinge": 0.0} and now.joints["panda_joint1"] == 0.6 and now.base[1] == pytest.approx(1.9)
