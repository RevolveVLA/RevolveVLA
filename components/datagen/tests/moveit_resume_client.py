"""Client of test_moveit_gpu.py's resume cases, run in a client container (revolve_common.sim_client): `<task> <robot> <instance>
<mode> <n>`, printing one RESULT JSON line with one row per episode.
  pregrasp: the sim starts at the pre-grasp: `n` episodes of the generator with `resume: true`.
  takeover: the base scene's start. First the pre-grasp state is probed (the expert drives to it, no episode), then `n` episodes in
            which the expert drives for 2 s, stops (a frozen policy takes over) and `resume()` finishes the task."""
import json
import sys
from pathlib import Path

from datagen.plugins.moveit import setup
from datagen.plugins.moveit.experts import EXPERTS
from datagen.plugins.moveit.generator import MoveItGenerator
from datagen.plugins.moveit.robot_io import EpisodeOver
from revolve_common.sim_client import run_client

REPO, TAKEOVER_S = "/repo", 2.0


class Stop(EpisodeOver):
    """Ends an expert's run where the test wants it."""


def row(client, n, notes, started) -> dict:
    result = client.wait_result(n)
    log = json.loads(notes) if notes.startswith("[") else []
    marks = {m["what"]: m for m in log}
    return {"episode": n, "passed": bool(result["passed"]), "failed_phase": result["failed_phase"], "reason": result["reason"],
            "entry": marks.get("resume", {}).get("entry"), "situation": marks.get("resume"),
            "whats": [m["what"] for m in log], "max_delta_action": marks.get("first_second", {}).get("max_delta_action"),
            "error": None if log else notes, "wall_s": round(client.now() - started, 1)}


def pregrasp(client, task, robot, instance, n):
    gen = MoveItGenerator(None, {"resume": True, "repo": REPO})
    gen.prepare(client, instance, task, robot)
    rows = []
    for i in range(n):
        if i:
            client.reset()
        k = client.start_episode()
        t0 = client.now()
        rows.append(row(client, k, gen.drive(client, instance, task, robot)["notes"], t0))
    return {"episodes": rows}


def probe(client, expert_cls, s) -> dict:
    """The robot at the expert's pre-grasp (base, arm, hand), reached by the expert itself; no episode runs."""
    class Probe(expert_cls):
        def mark(self, what, **kw):
            super().mark(what, **kw)
            if what == "pre_grasp":
                if not kw["ok"]:
                    raise RuntimeError("the pre-grasp was not reached")
                raise Stop()
    Probe(s).run()
    q = s.io.joints()
    return {"base": [round(v, 5) for v in s.io.base()[1]], "arm": [round(q[j], 5) for j in s.case["arm_joints"]]}


def takeover(client, task, robot, instance, n):
    s = setup.build(client, instance, task, robot, Path(REPO))
    start = probe(client, EXPERTS[task], s)
    rows = []
    for _ in range(n):
        client.reset()
        k = client.start_episode()
        t0 = client.now()
        io, check = s.io, s.io.check

        def handover():
            check()
            if client.now() - t0 >= TAKEOVER_S:
                raise Stop()
        io.check = handover
        EXPERTS[task](s).run()                              # the expert's first seconds (the policy's stand-in is its own pause)
        io.check = check
        rows.append(row(client, k, EXPERTS[task](s).resume(), t0))
    return {"pregrasp": start, "episodes": rows}


def main(client, task, robot, instance, mode, n):
    client.wait_ready()
    print("RESULT " + json.dumps({"pregrasp": pregrasp, "takeover": takeover}[mode](client, task, robot, instance, int(n))), flush=True)


if __name__ == "__main__":
    run_client(main, sys.argv[1], *sys.argv[2:])
