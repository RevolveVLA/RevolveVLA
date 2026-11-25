# training

Training fine-tunes a SmolVLA policy on a dataset with LeRobot 0.6.1's own `lerobot-train`, and serves a trained
checkpoint to the robot through Rosetta's policy server. It sits between dataset and evaluation in the loop. Each
job runs in the `revolve-training` container under the shared job wrapper, on the local GPU, on a remote GPU
machine over ssh, or on Modal.

Status on main: `launch.py`, the host side, fails at import. It loads `components/simulator/launch.py`, which was
removed with the old simulator. The image and the in-container `entry.py` do not need that file. Training also
still reads datasets through the old dataset component (`labels.py`, repo id `revolve/dataset`) and has not moved
to dataset v2 yet.

## Inputs

| Input | Comes from | Format |
|---|---|---|
| Dataset | the dataset component | a LeRobot v3.0 dataset directory |
| Base model | Hugging Face, cached in `cache/hf/` | default `lerobot/smolvla_base`, public, no token |
| Previous checkpoint | an earlier training job's `<out>/` | needs `config.json` and `model.safetensors` |
| Retrain mode | the caller | `finetune_last` (previous weights, full dataset), `from_base`, `finetune_recovery_only` (previous weights, recovery episodes only) |
| Run settings | the caller | `steps`, `batch_size`, `save_every`, `backend` (`local`, `gpu_machine`, `modal`) and its settings |
| GPU machine | `REVOLVE_GPU_MACHINE_HOST=user@host`, your `~/.ssh` | passwordless ssh; the job builds its own venv in a scratch dir there and removes it at the end |
| Modal credentials | `~/.modal.toml` | mounted read-only when it exists |
| Rosetta | `ThirdParty/rosetta` submodule | mounted read-only for serving |

## Outputs

| Output | Where it lands | Read by |
|---|---|---|
| Final checkpoint | `<out>/`: the `pretrained_model` files plus `train.json` (mode, init, episodes, steps, losses, model sha256) | evaluation, the next training turn |
| Earlier checkpoints | `<out>_<step>/`, only with `save_every` (backends `local` and `gpu_machine`) | evaluation |
| Job files | `<out parent>/jobs/train_<out name>/`: `job.json`, `status.json`, `result.json`, `stdout.log` | the manager |
| Live progress | `GET /status` on the job container, port 8000: step, loss, learning rate, ETA, GPU memory | the manager |
| Policy server | `<container name>:8080` on an internal Docker network only, because the server unpickles what clients send | evaluation rollouts |

## Run

```
./revolve build training      # the dependency image; nothing is built from source
```

Then from Python, once `launch.py` imports again:

```
launch.train(dataset, out, mode=M, steps=N, batch_size=B, backend="local")   # blocks until the job ends
job = launch.start_train(dataset, out, ...)                                   # same, without blocking
job = launch.start_serve(ckpt, job_dir, network)    # policy server; network from docker network create --internal
launch.stop(job)                                    # stop the server
```

Exit codes of a training job: those of `lerobot-train`, 3 when a Hugging Face token was found, 2 for bad arguments
or a missing checkpoint. On the reference laptop (RTX 5060, 8 GB) batch 2 used 1.9 GB at 0.15 s per step.

## Test

```
./revolve test training
```

`test_training_unit.py` needs no Docker: progress parsing, retrain plans, and the `gpu_machine` and `modal`
backends against fakes. On main 12 of its 14 tests pass; the other 2 fail on the missing simulator `launch.py`.
`test_training.py` runs real jobs on the GPU (40 steps, then serving one request) and today fails at collection
for the same reason. Its GPU machine part runs only when `REVOLVE_GPU_MACHINE_HOST` is set.

## Files

- `launch.py`: host side; starts training and serve jobs with the Docker SDK.
- `src/training/entry.py`: what the job container runs, `train` or `serve`.
- `src/training/train.py`: retrain mode to `lerobot-train` flags, and the final checkpoint to `<out>/`.
- `src/training/local.py`, `gpu_machine.py`, `modal_backend.py`: one module per backend.
- `requirements.lock`: the hashed pins for the image (LeRobot 0.6.1, torch 2.11.0 with CUDA 13.0).
