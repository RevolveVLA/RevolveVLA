"""Recovery tests: CPU by default (fake plugins, no docker, no ROS). Tests marked `gpu` need a free GPU, docker and the simulator
image and run with RECOVERY_GPU=1."""
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parents[1]
REPO = COMP.parents[1]
sys.path[:0] = [str(REPO / "common"), str(COMP / "src"), str(REPO / "components" / "datagen" / "src"),
                str(REPO / "components" / "evaluation" / "src"), str(COMP / "tests")]
sys.path.append(str(REPO / "components" / "datagen" / "tests"))        # fake_datagen: the fake training generator of an eval's train spec


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a free GPU, docker and the built images; set RECOVERY_GPU=1 to run")
