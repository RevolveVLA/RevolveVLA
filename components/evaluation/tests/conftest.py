"""Evaluation tests: CPU by default (fake plugins, no docker). Tests marked `gpu` need a free GPU, docker and the simulator image
and run with EVAL_GPU=1."""
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parents[1]
REPO = COMP.parents[1]
sys.path[:0] = [str(COMP / "src"), str(REPO / "components" / "datagen" / "src")]
sys.path.append(str(REPO / "components" / "datagen" / "tests"))      # fake_datagen: the fake training generator


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a free GPU, docker and the built images; set EVAL_GPU=1 to run")
