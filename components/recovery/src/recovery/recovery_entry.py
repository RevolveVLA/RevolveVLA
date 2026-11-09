"""Recovery entrypoint (runs inside revolve-recovery, normally under revolve_common.wrapper with
--parser recovery_progress:parse):

  recovery_entry.py plan --dir D --seed S --attempts N
      one plan_from_state run into D, which holds regen.json + start_state.npz + start_state_next.npz
      (recover.py writes them; everything else in D is removed first): the robot's gz models (the
      simulator's generator, runs/models, shared), then plan_from_state.py. Exit 0 = the run finished
      (handoff.json written; 0 planned is still 0), 3 = plan_from_state.py failed (no IK at all, a crash).

The gz side (actions, clearance precheck, the restored run, acceptance, the bag) runs in the simulator
image (regen_sim.py); the host composition is components/recovery/recover.py.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import entry as datagen_entry     # components/datagen/src/datagen (PYTHONPATH): call() in a process group

HERE = Path(__file__).resolve().parent
KEEP = ("regen.json", "start_state.npz", "start_state_next.npz")


def plan(a):
    out = datagen_entry.rpath(a.dir)
    for p in out.iterdir():
        if p.name not in KEEP:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    rg = json.loads((out / "regen.json").read_text())
    import cli as sim_cli                     # the simulator's model generator (components/simulator/src/sim)
    import spec as sim_spec
    datagen_entry.progress(stage="models", task=rg["task"], phase=rg["phase"], mode=rg["mode"], seed=a.seed)
    sim_cli.models(sim_spec.task(rg["task"]))
    datagen_entry.progress(stage="plan", attempts=a.attempts)
    rc = datagen_entry.call([sys.executable, str(HERE / "plan_from_state.py"), str(out)],
                            dict(os.environ, SEED=str(a.seed), ATTEMPTS=str(a.attempts)))
    if rc != 0 or not (out / "handoff.json").is_file():
        print(f"plan_from_state.py exited {rc}")
        return 3
    h = json.loads((out / "handoff.json").read_text())
    datagen_entry.progress(stage="done", planned=h["planned"], attempts=h["attempts"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--dir", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--attempts", required=True, type=int)
    a = ap.parse_args(argv)
    return {"plan": plan}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
