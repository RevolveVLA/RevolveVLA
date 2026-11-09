"""revolve-dataset entrypoint (run under revolve_common.wrapper by launch.py; Rosetta's install sourced):

  entry.py convert --task T --turn K --out DIR EPISODE_DIR...   episode dirs (bag/ + episode.json) -> dataset
  entry.py merge   --task T --out DIR DATASET...                 datasets -> one dataset (input order)

Both build <out>.partial/, load it with LeRobotDataset (dataset_info.py), check it (checks.py: frames vs
the episodes' sim durations, feature keys/shapes vs the robot's Rosetta contract, the task.yaml prompt,
labels), write meta/revolve_dataset.json (the summary + inputs) and only then move it to <out>. An <out>
whose revolve_dataset.json lists the same inputs is kept (resume). Last line: PROGRESS {"stage": "done", ...}.
Exit 1 with the problems listed when a check fails (<out>.partial is kept for inspection)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

import checks as C
import dataset_info
import labels as L

REPO = Path(os.environ.get("REPO", "/repo"))
SUMMARY = Path("meta") / "revolve_dataset.json"


def progress(**d):
    print("PROGRESS " + json.dumps(d), flush=True)


def task_facts(task):
    from revolve_common import specs
    r = specs.load(REPO / "specs" / "tasks" / task / "task.yaml")
    contract = REPO / "specs" / "robots" / r.robot.name / r.robot.contract
    return r.task.prompt, contract, yaml.safe_load(contract.read_text())


def finish(built: Path, out: Path, kind, inputs, sims, prompt, contract, want_labels, extra=None) -> int:
    s = dataset_info.summary(built)
    problems = C.dataset_problems(s, sims, contract["fps"], prompt, contract, want_labels) + (extra(s) if extra else [])
    rec = dict(kind=kind, inputs=[L.rel(p, REPO) for p in inputs], episodes=s["info"]["total_episodes"],
               frames=s["info"]["total_frames"], lengths=s["lengths"], tasks=s["tasks"], labels=s["labels"],
               features={k: v["shape"] for k, v in s["info"]["features"].items()}, problems=problems)
    (built / SUMMARY).write_text(json.dumps(rec, indent=1))
    if problems:
        progress(stage="failed", problems=problems, partial=L.rel(built, REPO))
        print("\n".join(["dataset check failed:", *problems]), file=sys.stderr)
        return 1
    if out.exists():
        shutil.rmtree(out)
    built.rename(out)
    shutil.rmtree(out.parent / f"{out.name}.partial")
    progress(stage="done", dataset=L.rel(out, REPO), episodes=rec["episodes"], frames=rec["frames"],
             episodes_converted=rec["episodes"], problems=[])
    return 0


def done_before(out: Path, inputs) -> bool:
    f = out / SUMMARY
    if f.is_file() and json.loads(f.read_text()).get("inputs") == [L.rel(p, REPO) for p in inputs]:
        rec = json.loads(f.read_text())
        progress(stage="done", dataset=L.rel(out, REPO), episodes=rec["episodes"], frames=rec["frames"],
                 episodes_converted=rec["episodes"], problems=[], resumed=True)
        return True
    return False


def convert(a) -> int:
    import port
    prompt, contract_path, contract = task_facts(a.task)
    eps, out = [Path(e) for e in a.episodes], Path(a.out)
    if done_before(out, eps):
        return 0
    progress(stage="port", episodes_converted=0, episodes_total=len(eps))
    built = port.port(eps, out, contract_path, a.turn, REPO)
    sims = [C.sim_s(json.loads((e / "episode.json").read_text())) for e in eps]
    want = [L.of_episode(e, a.turn, REPO) for e in eps]
    return finish(built, out, "convert", eps, sims, prompt, contract, want)


def merge(a) -> int:
    import merge as M
    prompt, _, contract = task_facts(a.task)
    ins, out = [Path(p) for p in a.datasets], Path(a.out)
    if done_before(out, ins):
        return 0
    parts = [json.loads((p / SUMMARY).read_text()) for p in ins]
    progress(stage="merge", datasets=len(ins), episodes_total=sum(p["episodes"] for p in parts))
    built = M.merge(ins, out)
    lengths = [n for p in parts for n in p["lengths"]]

    def same_lengths(s):
        return [] if s["lengths"] == lengths else [f"lengths: {s['lengths']} != the inputs' {lengths}"]
    return finish(built, out, "merge", ins, [None] * len(lengths), prompt, contract,
                  [lab for p in parts for lab in p["labels"]], same_lengths)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--task", required=True)
    c.add_argument("--turn", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("episodes", nargs="+")
    m = sub.add_parser("merge")
    m.add_argument("--task", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("datasets", nargs="+")
    a = ap.parse_args(argv)
    return convert(a) if a.cmd == "convert" else merge(a)


if __name__ == "__main__":
    sys.exit(main())
