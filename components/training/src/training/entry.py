"""Training component entrypoints, run under revolve_common.wrapper in revolve-training (launch.py):

  entry.py train --backend local|gpu_machine|modal --dataset D --out O --mode M --base B [--prev P]
                 --steps N --batch-size B [--seed S] [--log-freq F] [--settings JSON] [--job NAME]
      train.plan() (retrain mode -> init weights + episodes), the backend runs lerobot-train (its output is on
      our stdout, parsed by progress:parse), train.finalize() moves the final checkpoint to O (+ train.json).
  entry.py serve --ckpt C [--host H] [--port 8080] [--device cuda] [--fps 30]
      Rosetta's policy server (lerobot_rosetta.policy_server over LeRobot 0.6.1's async gRPC server),
      unchanged protocol. It unpickles what clients send (by design): launch.py only ever puts it on an
      internal docker network, no published port.
Exit codes: lerobot-train's; 3 a Hugging Face token was found; 2 bad arguments / no checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import progress
import train as T

BACKENDS = ("local", "gpu_machine", "modal")


def backend(name):
    if name == "local":
        import local as b
    elif name == "gpu_machine":
        import gpu_machine as b
    elif name == "modal":
        import modal_backend as b
    else:
        raise SystemExit(f"backend {name!r}: one of {BACKENDS}")
    return b


def cmd_train(a) -> int:
    t0 = time.time()
    try:
        p = T.plan(dataset=a.dataset, out=a.out, mode=a.mode, base=a.base, prev=a.prev, steps=a.steps,
                   batch_size=a.batch_size, seed=a.seed, log_freq=a.log_freq, num_workers=a.num_workers, job=a.job)
    except ValueError as e:
        print(f"train: {e}", flush=True)
        return 2
    T.progress(phase="plan", backend=a.backend, mode=p["mode"], init=p["init"], episodes=p["episodes"],
               total_steps=a.steps, log_freq=a.log_freq, batch_size=a.batch_size)
    progress.reset()
    progress.parse(f'PROGRESS {{"log_freq": {a.log_freq}}}')
    seen = {}

    def on_line(line):
        d = progress.parse(line)
        if d:
            seen.update(d)

    import shutil
    shutil.rmtree(a.out + ".partial", ignore_errors=True)
    res = backend(a.backend).run(p, json.loads(a.settings or "{}"), a.job, on_line)
    if res["rc"]:
        print(f"train: {a.backend} trainer exited {res['rc']}", flush=True)
        return res["rc"]
    info = T.finalize(p, seen, dict(backend=a.backend, wall_s=round(time.time() - t0, 1),
                                    **({"remote": res["remote"]} if res.get("remote") else {})), res.get("ckpt"))
    T.progress(phase="done", ckpt=a.out, model_sha256=info["model_sha256"], final_loss=seen.get("loss"))
    return 0


def cmd_serve(a) -> int:
    if not os.path.isfile(os.path.join(a.ckpt, "config.json")):
        print(f"serve: no checkpoint at {a.ckpt}", flush=True)
        return 2
    argv = [sys.executable, "-B", "-m", T.POLICY["serve"]["module"], "--host", a.host, "--port", str(a.port),
            "--fps", str(a.fps), "--pretrained-name-or-path", a.ckpt, "--policy-device", a.device]
    print("serve: " + " ".join(argv), flush=True)
    os.execv(argv[0], argv)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--backend", choices=BACKENDS, default="local")
    t.add_argument("--dataset", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--mode", choices=T.MODES, default="finetune_last")
    t.add_argument("--base", default="lerobot/smolvla_base")
    t.add_argument("--prev", default=None)
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--batch-size", type=int, required=True)
    t.add_argument("--seed", type=int, default=1000)
    t.add_argument("--log-freq", type=int, default=10)
    t.add_argument("--num-workers", type=int, default=4)
    t.add_argument("--settings", default=None, help="the backend's settings (experiment training.backends.<b>) as JSON")
    t.add_argument("--job", default="train")
    s = sub.add_parser("serve")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=T.POLICY["serve"]["port"])
    s.add_argument("--device", default=T.POLICY["serve"]["device"])
    s.add_argument("--fps", type=int, default=30)
    a = ap.parse_args(argv)
    return cmd_train(a) if a.cmd == "train" else cmd_serve(a)


if __name__ == "__main__":
    sys.exit(main())
