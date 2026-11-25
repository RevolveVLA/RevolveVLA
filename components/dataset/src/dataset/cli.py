"""The dataset component's entrypoint (DATASET-v2.md section 7), in the dataset image with Rosetta's install sourced:

  python3 -m dataset.cli convert --manifest M [--manifest M2 ...] [--split S] --turn K --out D
  python3 -m dataset.cli merge --out D DATASET [DATASET ...]

`./revolve dataset ...` runs it in a container (launch.py, which parses argv with the same parser()); the HTTP job
wrapper runs the same command. Both build <D>.partial/, check it (checks.py) and only then move it to <D>.

The out check comes first: <D> must be missing, empty or a dataset of this component (record.py), <D>.partial missing,
empty or a build directory of this component, and neither may be or hold an input; else the command is refused and
nothing is touched. A finished dataset at <D> whose record lists the same inputs is kept (resume, "resumed": true);
anything else at <D> is removed once resume is ruled out, so after any failure past the out check nothing is at <D>.

Stages: out check, inputs (manifests, task, contract, results, bags; nothing ported yet), build, checks. A stage that
refuses raises Refused; one handler per command reports it. Progress: `PROGRESS {json}` lines (stage checks | port |
merge | done | failed, episodes_converted, episodes_total). The last stdout line is the result {dataset, episodes,
frames, problems} (+ "warnings" when there are any). Exit 0 only when the dataset is at <D> and every check passed; on
any problem exit 1 with the problems listed (<D>.partial/ kept for inspection after a build). The out and input checks
run before anything heavy is imported, except reading the bags' time lines (rosbag2)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from . import checks, contract as C, manifest as M, merge as G, record as R
from .layout import CONTRACT, PARTIAL_MARK, partial, progress

REPO = Path(os.environ.get("REVOLVE_REPO") or Path(__file__).resolve().parents[4])
CHECKS = {"convert": ("port", "frames", "features", "task", "labels", "episodes"),
          "merge": ("merge", "frames", "features", "task", "labels", "episodes")}


class Refused(Exception):
    """A stage's problems: the command stops there."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def refuse(problems) -> None:
    if problems:
        raise Refused(problems)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p) -> str:
    """Repo paths (container /repo or the host checkout) repo-relative; anything else as is."""
    p = Path(p)
    for root in (REPO, Path("/repo")):
        if p.is_absolute() and p.is_relative_to(root):
            return str(p.relative_to(root))
    return str(p)


def task_facts(task, robot):
    """-> (task.yaml's prompt, the robot's contract path); Refused when either is missing."""
    from revolve_common import specs
    try:
        prompt = specs.resolve(REPO / "specs", task, robot=robot, assets=False).task.prompt
    except specs.SpecError as e:
        raise Refused([f"inputs: task {task!r} with robot {robot!r}: {str(e).splitlines()[0]}"]) from None
    path = REPO / "specs" / "robots" / robot / specs.CONTRACT_FILE
    if not path.is_file():
        raise Refused([f"inputs: robot {robot!r} has no contract {rel(path)}"])
    return prompt, path


def claim_out(out: Path, inputs) -> list[str]:
    """Problems when <out> or <out>.partial must not be replaced: it is or holds an input, or it holds something that
    is not this component's (a dataset with a record at <out>, a build directory at <out>.partial)."""
    o = out.absolute().resolve()
    work = partial(o)
    problems = []
    for d in (o, work):
        hit = [str(p) for p in inputs if (q := Path(p).absolute().resolve()) == d or q.is_relative_to(d)]
        if hit:
            problems.append(f"out: {d} is or holds inputs {hit[:3]}{' ...' if len(hit) > 3 else ''}; refusing to "
                            f"replace it")
    for d, ours, what in ((o, lambda d: R.read(d) is not None, "a dataset of this component (no readable record)"),
                          (work, lambda d: (d / PARTIAL_MARK).is_file(), "a build directory of this component")):
        if d.exists() and not d.is_dir():
            problems.append(f"out: {d} exists and is not a directory")
        elif d.is_dir() and any(d.iterdir()) and not ours(d):
            problems.append(f"out: {d} exists and is not {what}; refusing to replace it")
    return problems


def clear_out(out: Path) -> None:
    """Remove what claim_out allowed at <out>: an empty directory or a dataset of this component."""
    if out.is_dir():
        shutil.rmtree(out)


def done(kind, out: Path, episodes, frames, problems, warnings=(), **extra) -> int:
    ok = not problems
    extra = dict(extra, **({"warnings": list(warnings)} if warnings else {}))
    progress(stage="done" if ok else "failed", dataset=str(out), episodes=episodes, frames=frames,
             episodes_converted=episodes if ok else 0, problems=problems, **extra)
    if problems:
        print("\n".join([f"dataset {kind} failed:", *problems]), file=sys.stderr)
    print(json.dumps(dict(dataset=str(out), episodes=episodes, frames=frames, problems=problems, **extra)), flush=True)
    return 0 if ok else 1


def run(kind, out: Path, sources, body) -> int:
    """The out check, then body(): a stage that raises Refused, or any other exception, clears <out> and reports."""
    problems = claim_out(out, sources)
    if problems:
        return done(kind, out, 0, 0, problems)
    try:
        return body()
    except Refused as r:
        problems = r.problems
    except Exception as e:      # a bug, or an input no check foresaw: still one result line, nothing at <out>
        problems = [f"internal: {type(e).__name__}: {e}"]
    clear_out(out)
    return done(kind, out, 0, 0, problems)


def resume(out: Path, kind, inputs) -> bool:
    """True when <out> is a finished dataset of these inputs (it is kept); else <out> is cleared for the build."""
    rec = R.read(out)
    if rec is not None and rec.ok and rec.kind == kind and rec.inputs == inputs:
        return True
    clear_out(out)
    return False


def fresh_work(out: Path) -> Path:
    work = partial(out)
    shutil.rmtree(work, ignore_errors=True)      # a stale build: LeRobot's writer refuses an existing directory
    work.mkdir(parents=True)
    (work / PARTIAL_MARK).touch()
    return work


def finish(kind, out: Path, work: Path, built: Path | None, rec: R.Record, problems: list[str], warnings=()) -> int:
    rec.checks = {c: ("failed" if any(p.startswith(c + ":") for p in problems) else "ok") for c in CHECKS[kind]}
    rec.warnings, rec.problems = list(warnings), problems
    if built is not None and built.is_dir():
        rec.write(built)
    if problems:
        return done(kind, out, rec.episodes, rec.frames, problems, warnings, partial=str(work))
    built.rename(out)
    shutil.rmtree(work)
    return done(kind, out, rec.episodes, rec.frames, [], warnings)


def load_summary(root, problems):
    from .summary import summary
    try:
        return summary(root)
    except Exception as e:      # the dataset does not load with LeRobot: a problem, never a crash
        problems.append(f"episodes: {root} does not load with LeRobotDataset ({type(e).__name__}: {e})")
        return None


def out_path(arg) -> Path:
    """--out, absolute with symlinks resolved: the one path of every check and filesystem action on it."""
    return Path(arg).absolute().resolve()


def convert(a) -> int:
    out = out_path(a.out)
    plan = M.read(a.manifest, a.split, a.turn)
    sources = [Path(m).parent for m in a.manifest] + [p for e in plan.episodes for p in (e.bag, e.result)]

    def body():
        refuse(plan.problems)
        prompt, cpath = task_facts(plan.task, plan.robot)
        inputs = dict(manifests=list(a.manifest), manifest_sha256=plan.digests, split=a.split, turn=a.turn,
                      contract=rel(cpath), contract_sha256=sha256(cpath), prompt=prompt)
        if resume(out, "convert", inputs):
            rec = R.read(out)
            return done("convert", out, rec.episodes, rec.frames, [], rec.warnings, resumed=True)
        eps, contract = plan.episodes, C.load(cpath)
        fps = contract["fps"]
        progress(stage="checks", episodes_converted=0, episodes_total=len(eps))
        refuse(M.check_inputs(eps, C.topics(contract), plan.task, plan.robot))
        from . import port, ticks
        bag_ticks, problems = ticks.of_episodes(eps, cpath, fps)
        refuse(problems)

        work = fresh_work(out)
        progress(stage="port", episodes_converted=0, episodes_total=len(eps))
        built, saved, problems = port.port([e.bag for e in eps], [e.labels for e in eps], cpath, work,
                                           names=[e.where for e in eps])
        progress(stage="checks", episodes_converted=len(saved), episodes_total=len(eps))
        s = load_summary(built, problems) if built.is_dir() else None
        lengths = s["lengths"] if s else saved
        warnings = []
        for e, f, t in zip(eps, lengths, bag_ticks, strict=False):
            p, w = checks.convert_episode(e.where, f, e.duration_s, t, fps)
            problems, warnings = problems + p, warnings + w
        if s is not None:
            problems += checks.dataset_problems(s, fps=fps, prompt=prompt, features=C.features(contract),
                                                labels=[e.labels for e in eps])
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr, flush=True)
        rec = R.Record(
            kind="convert", inputs=inputs, contract=rel(cpath), prompt=prompt, episodes=len(lengths),
            frames=sum(lengths), features=checks.features_of(s["info"]) if s else {},
            frame_tolerance=checks.FRAME_TOLERANCE, action_late_fails=checks.ACTION_LATE_FAILS,
            per_episode=[dict(manifest_entry=e.where, bag=str(e.bag), result=str(e.result), duration_s=e.duration_s,
                              frames=f, offset=checks.frame_offset(f, e.duration_s, fps), bag_ticks=t.ticks,
                              warmup_ticks=t.warmup, action_start_frames=t.action_late, labels=e.labels)
                         for e, f, t in zip(eps, lengths, bag_ticks, strict=False)])
        return finish("convert", out, work, built, rec, problems, warnings)

    return run("convert", out, sources, body)


def merge(a) -> int:
    out, ins = out_path(a.out), [Path(p) for p in a.datasets]

    def body():
        recs, problems = G.inputs(ins)
        refuse(problems)
        inputs = dict(datasets=list(a.datasets), summary_sha256=[sha256(p / R.SUMMARY) for p in ins])
        if resume(out, "merge", inputs):
            rec = R.read(out)
            return done("merge", out, rec.episodes, rec.frames, [], rec.warnings, resumed=True)
        refuse(G.incompatible(ins))
        problems = []
        parts = [load_summary(p, problems) for p in ins]
        refuse(problems)
        contract = yaml.safe_load((ins[0] / CONTRACT).read_text())      # the inputs' own contract
        prompt = recs[0].prompt
        labels = [lab for s in parts for lab in s["labels"] or []]
        lengths = [n for s in parts for n in s["lengths"]]
        names = [f"{p} episode {i}" for p, s in zip(ins, parts, strict=True) for i in range(len(s["lengths"]))]
        work = fresh_work(out)
        progress(stage="merge", datasets=len(ins), episodes_converted=0, episodes_total=len(lengths))
        built = None
        try:
            built = G.merge(ins, work)
        except Exception as e:
            problems.append(f"merge: aggregate_datasets failed ({type(e).__name__}: {e})")
        s = load_summary(built, problems) if built else None
        if s is not None:
            problems += [p for n, f, want in zip(names, s["lengths"], lengths, strict=False)
                         for p in checks.merge_episode(n, f, want)]
            problems += checks.dataset_problems(s, fps=contract["fps"], prompt=prompt,
                                                features=C.features(contract), labels=labels)
        rec = R.Record(kind="merge", inputs=inputs, contract=recs[0].contract, prompt=prompt, episodes=len(lengths),
                       frames=sum(lengths), features=checks.features_of(s["info"]) if s else {},
                       per_episode=[dict(source=n, frames=f, labels=lab)
                                    for n, f, lab in zip(names, lengths, labels, strict=False)])
        return finish("merge", out, work, built, rec, problems)

    return run("merge", out, ins, body)


def parser() -> argparse.ArgumentParser:
    """The entrypoint's arguments; launch.py parses with the same parser. No abbreviated options: launch must see
    exactly the paths the entrypoint will use."""
    ap = argparse.ArgumentParser(prog="dataset", description=__doc__.split("\n\n")[0], allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", allow_abbrev=False, help="the passed episodes of producer manifests -> one dataset")
    c.add_argument("--manifest", action="append", required=True,
                   help="a producer's manifest.json (repeatable, in order)")
    c.add_argument("--split", help="only the entries of this split")
    c.add_argument("--turn", type=int, required=True, help="the `turn` label of every episode")
    c.add_argument("--out", required=True, help="the dataset directory (built in <out>.partial/)")
    m = sub.add_parser("merge", allow_abbrev=False, help="finished datasets -> one dataset, in the given order")
    m.add_argument("--out", required=True)
    m.add_argument("datasets", nargs="+")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    return convert(a) if a.cmd == "convert" else merge(a)


if __name__ == "__main__":
    sys.exit(main())
