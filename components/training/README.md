# training
SmolVLA training (LeRobot 0.6.1's own `lerobot-train`) and serving (Rosetta's `lerobot_rosetta.policy_server`
over LeRobot's async gRPC server), each under the revolve_common HTTP wrapper. Backends: local, gpu_machine, modal.

| Path | What |
|---|---|
| `Dockerfile`, `build.sh` | revolve-training: deps only (debian slim, ffmpeg, uv 0.9.15, uv-managed Python 3.12.12, `uv pip sync --require-hashes requirements.lock`, + pydantic/FastAPI/uvicorn/modal/ssh/rsync with the lock as a constraint). `./revolve build training` |
| `requirements.lock` / `.in` | copy of the earlier pipeline's serve lock (= the training lock + LeRobot's async extra, same pins): one image for train and serve |
| `launch.py` | host side (docker SDK): `start_train` / `train` / `train_from_spec(training, dataset, out, prev)`, `start_serve(ckpt, job_dir, internal_network)`, `stop`, `in_image` |
| `src/training/entry.py` | `train --backend ...` and `serve` (the wrapper's command) |
| `src/training/train.py` | retrain modes -> lerobot-train flags (`plan`), the final checkpoint -> `<out>/` + train.json (`finalize`), `stream` (process group + SIGTERM) |
| `src/training/local.py`, `gpu_machine.py`, `modal_backend.py` | one module per backend, same `run(plan, settings, job, on_line)` contract |
| `src/training/progress.py` | wrapper parsers: `parse` (lerobot-train log line + tqdm bar: step, total_steps, loss, lr, grad_norm, s/step, eta_s, gpu_mem_gb, losses) and `parse_serve` (ready, chunks) |
| `src/training/smolvla.yaml` | the legacy policy plugin's train_args (unchanged) |
| `src/training/serve_check.py` | one inference request as LeRobot's RobotClient makes it (tests) |

## Job contract
`train(dataset, out, mode, base, prev, steps, batch_size, backend, settings)`: job dir `<out parent>/jobs/train_<out
name>/` (job.json, status.json, result.json, stdout.log). `out/` = the final checkpoint's pretrained_model files +
train.json (mode, init, episodes, step, model_sha256, losses, backend, remote timings). Modes (experiment
training.mode): `finetune_last` (previous checkpoint's weights, full merged dataset; base on turn 0), `from_base`,
`finetune_recovery_only` (previous weights, `--dataset.episodes` = the dataset component's
`labels.select(source="recovery")`). Weights carry over through `policy.pretrained_path` (fresh optimizer and
schedule, as legacy; never `--resume`). steps / batch_size from the yaml; save only the final checkpoint, unless
`training.save_every: N` (-> `--save_freq`): then each earlier checkpoint's pretrained_model files are copied to
`<out>_<step>/` (e.g. `turn_0/ckpt_010000/`) as soon as LeRobot has written it, while the training goes on
(backends local and gpu_machine; not modal).

## Backends
- local: the container gets the GPU. RTX 5060 8 GB: batch 2 uses 1.9 GB, 0.15 s/step.
- gpu_machine (`{host, scratch: ~/revolve_scratch, gpu: 0}`): the host is `user@host` with passwordless ssh, taken
  from the environment variable `REVOLVE_GPU_MACHINE_HOST` when the experiment yaml gives no `host` (the shipped
  experiments do not); unset, a gpu_machine train step fails with an error naming the variable. Set it before
  `./revolve run` (e.g. `export REVOLVE_GPU_MACHINE_HOST=me@my-gpu-box`); the manager passes every `REVOLVE_*`
  variable on to its steps. The gpu_machine test is skipped when it is unset. A CPU container drives ssh/rsync (the user's ~/.ssh,
  /etc/passwd, /etc/group mounted read-only). Per job `<scratch>/<job>/` holds uv + the lock, a venv built there,
  every cache and HOME, the dataset and the init checkpoint; lerobot-train runs as an ssh command whose output
  streams back to the local wrapper (same parser, same /status); the final checkpoint is rsync'd back; then the
  job's processes are killed and the dir removed (and the scratch dir if empty), whatever happened.
- modal (`{gpu: L4, timeout_s, volume}`): one attached Function call (image from the same lock, no secrets),
  dataset/init uploaded to a Volume under jobs/<job>/, trainer output streamed back, checkpoint downloaded,
  jobs/<job>/ removed. Needs Modal's own credentials (~/.modal.toml, mounted read-only); unit-tested with a fake
  SDK only (no credentials on this host).

## Serve
`start_serve(ckpt, job_dir, network)`: refuses unless `network` is a `docker network create --internal` network
(the server unpickles client data); clients reach `<name>:8080`; no host ports. HF offline
(SmolVLM2 processor files from cache/hf). Status: `ready` after "RosettaPolicyServer started on", `chunks`.

## Tests
`./revolve test training` (conftest.py: fixtures; test_training_unit.py: no docker; test_training.py: real jobs).
