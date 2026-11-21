from glob import glob

from setuptools import find_packages, setup

setup(
    name="revolve_sim",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/revolve_sim"]),
        ("share/revolve_sim", ["package.xml"]),
        ("share/revolve_sim/launch", glob("launch/*.launch.py")),
        ("share/revolve_sim/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": ["world = revolve_sim.world:main", "evaluator = revolve_sim.evaluator:main",
                                      "reset = revolve_sim.reset:main"]},
)
