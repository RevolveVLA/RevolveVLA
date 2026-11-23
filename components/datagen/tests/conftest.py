"""Datagen tests: CPU by default (fake plugins, no docker). Tests marked `gpu` need a free GPU, docker and the simulator image
and run with DATAGEN_GPU=1."""
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COMP), str(COMP / "src")]


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a free GPU, docker and the built images; set DATAGEN_GPU=1 to run")
