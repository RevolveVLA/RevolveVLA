"""Datagen entrypoint (runs inside revolve-datagen, normally under revolve_common.wrapper with
--parser progress:parse):

  entry.py plan --task T --seed S --out D [--attempts N] [--centre-seed C]
      one AutoMoMa plan run into D (removed first): the robot's gz models (the simulator's generator,
      runs/models, shared), plan.py with the task's starts block (generator.yaml -> ROUND_STARTS), then
      fk_check.py (planner FK against the gz model). Exit 0 = planned and the FK check passed (as the
      the earlier pipeline's planner adapter), 1 = planned but the FK check failed (kept: the
      same seed plans the same), 3 = plan.py failed (retried on a resume).

The replay, acceptance and recording of the planned attempts run in the simulator component; the host
composition is components/datagen/generate.py.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import specs_bridge

HERE = Path(__file__).resolve().parent


def progress(**kw):
    print("PROGRESS " + json.dumps(kw), flush=True)


def rpath(p):
    p = Path(p)
    return p if p.is_absolute() else specs_bridge.ROOT / p


def call(cmd, env):
    """cmd in its own process group, which is killed after it exits: plan.py's nvidia-smi sampler outlives a
    crashed plan.py and would hold the wrapper's stdout pipe open."""
    p = subprocess.Popen(cmd, env=env, start_new_session=True)

    def kill(sig):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, lambda *_: (kill(signal.SIGTERM), sys.exit(143)))   # the wrapper's cancel
    try:
        return p.wait()
    finally:
        kill(signal.SIGKILL)


def plan(a):
    out = rpath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    import cli as sim_cli                     # the simulator's model generator (components/simulator/src/sim)
    import spec as sim_spec
    progress(stage="models", task=a.task, seed=a.seed)
    sim_cli.models(sim_spec.task(a.task))
    gen = specs_bridge.load("generator", "automoma")
    env = dict(os.environ, SEED=str(a.seed), ATTEMPTS=str(a.attempts or gen["attempts"]),
               CENTRE_SEED=str(a.seed if a.centre_seed is None else a.centre_seed))
    env.update({f"ROUND_{k.upper()}": json.dumps(v) for k, v in specs_bridge.round_blocks(a.task).items()
                if k == "starts"})
    progress(stage="plan", attempts=int(env["ATTEMPTS"]))
    rc = call([sys.executable, str(HERE / "plan.py"), str(out), f"task={a.task}"], env)
    if rc != 0 or not (out / "handoff.json").is_file():
        print(f"plan.py exited {rc}")
        return 3                      # no plan run (a crash, CUDA OOM, no IK at all): may be retried
    planned = json.loads((out / "handoff.json").read_text())["planned"]
    progress(stage="fk_check", planned=planned)
    rc = call([sys.executable, str(HERE / "fk_check.py"), str(out)], env)
    progress(stage="done", planned=planned, fk_passed=rc == 0)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--task", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--attempts", type=int, default=None, help="default: generator.yaml attempts")
    p.add_argument("--centre-seed", type=int, default=None, help="starts block centre seed (default: --seed)")
    a = ap.parse_args(argv)
    return {"plan": plan}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
