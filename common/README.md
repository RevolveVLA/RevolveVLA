# common (revolve_common)
Shared by every component and the manager. Small on purpose; built on pydantic, PyYAML, numpy, FastAPI + uvicorn.

| Module | What |
|---|---|
| `specs.py` | pydantic v2 schemas for specs/ (experiment, task, robot, scene) and `load(path)`: validates, checks seed ranges are disjoint, resolves experiment -> task -> robot + scene, and the recovery phases against the task's |
| `metrics.py` | metric registry: pure functions over a state dict (`REGISTRY`), `evaluate(condition, state or series)` with `hold_s` (every sample of the last hold_s), `offline:` thresholds and `only:` (offline / plan) |
| `job.py` | job contract: job.json in, status.json / result.json / stdout.log out, atomic writes |
| `gpu.py` | GPU preflight: `check()` (host nvidia-smi), `check_docker(dk, image)` (nvidia-smi in a throwaway container): GpuError on a nonzero exit, `ERR!`, "requires reset", a hang, or a GPU container docker cannot start |
| `containers.py` | `run(dk, image, cmd, **kw)`: docker SDK run that removes a container docker created but could not start |
| `wrapper.py` | `python -m revolve_common.wrapper --job-dir D [--port] [--parser mod:fn] [-- cmd]`: runs cmd, parses progress (default `PROGRESS {json}` lines), GET /status, GET /result, POST /cancel, exits with cmd's code |

## Setup (host)
```
cache/bin/uv venv .venv --python 3.12        # any uv works; .venv/ is git-ignored
VIRTUAL_ENV=.venv uv pip install -e "common[test]"
./revolve test common
./revolve validate specs/experiments/fridge_mini.yaml
```
`./revolve` re-runs itself with .venv's Python and without the host PYTHONPATH (the ROS one breaks pytest).
