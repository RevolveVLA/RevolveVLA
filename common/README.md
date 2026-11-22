# common (revolve_common)
Shared by every component and the manager. Small on purpose; built on pydantic, PyYAML, numpy, FastAPI + uvicorn.

| Module | What |
|---|---|
| `specs.py` | pydantic v2 schemas for specs/ (scene, robot, task, the sanity base tasks/sanity.yaml), the metric registry (`METRICS`: names, units, parameters), `load(path)` (validates; scene model:// names, joints and task entities checked against the model SDFs; `assets=False`: names only) and `resolve(root, task, robot, scene=)` (the task + scene + merged sanity checks, robot::<alias> and the robot start checked against the robot; `scene=`: a loaded scene instance instead of the task's scene file). What a simulator run would refuse: `check_run(scene, robot, task)` (used by `./revolve sim` and `validate`), from `scene_file` (a --scene value -> its file), `check_sim_support`, `check_robot_start` / `arm_errors`. Format: the module docstring |
| `job.py` | job contract: job.json in, status.json / result.json / stdout.log out, atomic writes |
| `gpu.py` | GPU preflight: `check()` (host nvidia-smi), `check_docker(dk, image)` (nvidia-smi in a throwaway container): GpuError on a nonzero exit, `ERR!`, "requires reset", a hang, or a GPU container docker cannot start |
| `containers.py` | `run(dk, image, cmd, **kw)`: docker SDK run that removes a container docker created but could not start |
| `wrapper.py` | `python -m revolve_common.wrapper --job-dir D [--port] [--parser mod:fn] [-- cmd]`: runs cmd, parses progress (default `PROGRESS {json}` lines), GET /status, GET /result, POST /cancel, exits with cmd's code |

## Setup (host)
```
cache/bin/uv venv .venv --python 3.12        # any uv works; .venv/ is git-ignored
VIRTUAL_ENV=.venv uv pip install -e "common[test]"
./revolve test common
./revolve validate                           # every spec under specs/, against assets/
```
`./revolve` re-runs itself with .venv's Python and without the host PYTHONPATH (the ROS one breaks pytest).
