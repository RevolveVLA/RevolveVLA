# common (revolve_common)

`revolve_common` is the small Python library every component shares. It loads and checks the specs, defines the
job files a component writes, wraps a component's tool in a small HTTP server, checks GPU health, and starts and
drives simulators for datagen and evaluation. It depends only on pydantic, PyYAML, numpy, FastAPI and uvicorn.

## Inputs

| Input | Comes from | Format |
|---|---|---|
| Specs | `specs/scenes/`, `specs/robots/`, `specs/tasks/` (with `sanity.yaml`), or a scene instance outside `specs/` | YAML; unknown keys, duplicate keys and NaN are errors |
| Robot contract | `specs/robots/<robot>/contract.yaml` | Rosetta contract, checked against the simulator's recorded topics and the robot's controllers |
| Scene assets | `assets/`, made by `./revolve assets` | SDF models; needed to check model, joint and link names |
| Job | `<job dir>/job.json`, written by whoever starts the job | the command to run and its parameters |
| Environment | your shell | `REVOLVE_GPU_PREFLIGHT=0` turns the GPU checks off; `REVOLVE_SIM_IMAGE` picks the simulator image |

## Outputs

| Output | Where it lands | Read by |
|---|---|---|
| Checked specs | Python objects from `specs.load` and `specs.resolve`, or a `SpecError` naming the bad key | the CLI, simulator, datagen, evaluation |
| Job status and result | `<job dir>/status.json`, `result.json`, `stdout.log`, each written atomically | the manager, you |
| Job HTTP server | `GET /status`, `GET /result`, `POST /cancel` on port 8000 | the manager |
| Sim sessions | a running `./revolve sim` plus a client container beside it, relaunched on known faults | datagen, evaluation |

## Run

One-time setup on the host:

```
uv venv .venv --python 3.12                       # .venv/ is git-ignored
VIRTUAL_ENV=.venv uv pip install -e "common[test]"
./revolve validate                                # every spec under specs/, against assets/
./revolve validate --no-assets                    # names and syntax only
```

`./revolve` re-runs itself with the `.venv` Python and drops the host's `PYTHONPATH`, because the ROS one breaks
pytest.

## Test

```
./revolve test common         # about 15 s on the CPU, no Docker
```

178 tests pass. 7 more need `assets/` and are skipped without it.

## Files

- `specs.py`: the spec schemas, the metric list (`METRICS`) and the loader. The format is in its docstring.
- `job.py`: the job files (`job.json` in; `status.json`, `result.json` and `stdout.log` out).
- `wrapper.py`: `python -m revolve_common.wrapper --job-dir D -- CMD` runs a tool and serves its progress over HTTP.
- `gpu.py`: GPU health checks that fail fast on a GPU that needs a reset.
- `containers.py`: a Docker SDK `run` that removes a container Docker created but could not start.
- `sim_session.py`, `sim_client.py`: the host side and the client side of one running simulator.
- `sim_fake.py`: CPU stand-ins for both, for tests.
- `policy_io.py`: the policy's observation and action layout; today only the dataset tests use it.
