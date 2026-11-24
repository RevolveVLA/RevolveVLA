"""The robots' Rosetta contracts (specs/robots/<r>/contract.yaml) loaded by Rosetta's own loader in the simulator image,
no sim (check_rosetta.py): every contract loads, its streams are the agreed ones (components/dataset/DATASET-v2.md)
and specs.check_contract passes with the cameras from the robot's description. The bags: test_contract.py."""
import os
import time

import robot_cases
from conftest import REPO, rosetta

IMAGES = ["observation.images.overhead_left", "observation.images.overhead_right", "observation.images.wrist"]
WIDTH = {"mobile_panda": 11, "mobile_so101": 9, "mobile_so101_xl": 9}


def test_contracts_load_with_rosetta():
    logs = os.environ.get("REVOLVE_TEST_LOGS", REPO / "runs" / "test_simulator" / time.strftime("%Y%m%d_%H%M%S"))
    log = REPO / logs / "test_rosetta_contracts" / "check_rosetta.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    r = rosetta(["--robots", *robot_cases.ROBOTS], log)
    for robot, c in r["contracts"].items():
        assert c["robot_type"] == robot and c["fps"] == 30 and c["tasks"] == ["/task/prompt"], c
        assert c["images"] == IMAGES and c["state"] == c["action"] == WIDTH[robot], c
        assert not c["check_contract"] and not c["warnings"], c
        for key, topic, timeline, dim, dtype, size in c["streams"]:
            assert timeline == ("receive" if key == "action" else "header"), (key, topic, timeline)
            if key in IMAGES:
                assert size == [480, 480] and dtype == "video" and topic.endswith("_cam/image/compressed"), (key, topic)
            else:
                assert dtype == "float64", (key, topic, dtype)
